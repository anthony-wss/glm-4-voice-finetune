# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_dataset
import pickle

from speech_tokenizer.utils import extract_speech_token
from src.model import FineTunedGLM4Voice, GLM4VoiceTokenizer, prepare_model_for_lora
from src.dataset import TextDataset
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import WhisperFeatureExtractor, AutoTokenizer
import os
from transformers import Trainer, TrainingArguments

os.environ["WANDB_PROJECT"] = "glm-4-voice-finetune"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

model_path = "THUDM/glm-4-voice-9b"
tokenizer_path = "THUDM/glm-4-voice-tokenizer"
device = "cuda"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.dataset[idx], return_tensors="pt", padding=True, truncation=True, max_length=512)
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids']
        }

class CustomDataCollator:
    """Custom data collator for batching audio and text inputs."""

    def __init__(self):
        pass

    def __call__(self, batch):
        return {
            "input_ids": torch.cat([item["input_ids"] for item in batch]),
            "attention_mask": torch.cat([item["attention_mask"] for item in batch]),
            "labels": torch.cat([item["labels"] for item in batch])
        }

glm_model = FineTunedGLM4Voice(model_path).model
# glm_model = prepare_model_for_lora(glm_model)
glm_tokenizer = GLM4VoiceTokenizer(model_path).tokenizer
print("Model loaded")

with open("train_texts.pkl", "rb") as f:
    train_texts = pickle.load(f)

train_dataset = CustomDataset(train_texts, glm_tokenizer)
print("Dataset loaded")

training_args = TrainingArguments(
    output_dir="test_trainer",
    eval_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_strategy="steps",
    logging_steps=1,
    max_steps = 5000,
    bf16=True,
    gradient_checkpointing=False,
    deepspeed="ds_config.json",
    label_names=["labels"]
)
trainer = Trainer(
    model=glm_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    data_collator=CustomDataCollator()
)
trainer.train()
