from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from speech_tokenizer.utils import extract_speech_token

from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import WhisperFeatureExtractor, AutoTokenizer
import torch
import os

os.environ["WANDB_PROJECT"] = "glm-4-voice-finetune"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


model_path = "THUDM/glm-4-voice-9b"
tokenizer_path = "THUDM/glm-4-voice-tokenizer"
device = "cuda"

whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
glm_model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=None,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

audio_path = "./input-sample-0.wav"
audio_tokens = extract_speech_token(
    whisper_model, feature_extractor, [audio_path]
)[0]  # 12.5 TPS
audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
user_input = audio_tokens
system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>streaming_transcription\n"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from typing import Dict, List, Optional
import logging
import os


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        # Create input_ids and labels
        self.input_ids = self.encodings['input_ids']
        self.attention_mask = self.encodings['attention_mask']
        # For causal language modeling, labels are the same as input_ids
        self.labels = self.input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

train_texts = [prompt] * 100

train_dataset = TextDataset(train_texts, glm_tokenizer)


def prepare_model_for_lora(model, tokenizer):
    # Configure LoRA specifically for ChatGLM architecture
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "query_key_value",  # Attention projection
            "dense",            # Attention output
            "dense_h_to_4h",   # MLP
            "dense_4h_to_h"    # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )

    # Prepare model for k-bit training if needed
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


glm_model = prepare_model_for_lora(glm_model, glm_tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduced batch size
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Increased for effective batch size
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    evaluation_strategy="no",
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=True,  # Use fp16 training
    gradient_checkpointing=True,
    # Memory optimizations
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
)

trainer = Trainer(
    model=glm_model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()


