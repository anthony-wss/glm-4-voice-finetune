# Load model directly
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from src.model import prepare_model_for_lora
import os
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer

os.environ["WANDB_PROJECT"] = "glm-4-voice-finetune"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

MAX_LENGTH = 2048

train_dataset = load_dataset("anthony-wss/rpg-overlap-30-35-processed", split="train")
eval_dataset = load_dataset("anthony-wss/rpg-overlap-30-35-processed", split="test")
print("Dataset loaded")

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

sft_config = SFTConfig(
    output_dir="output",
    max_seq_length=MAX_LENGTH,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_strategy="steps",
    logging_steps=1,
    max_steps = 5000,
    bf16=True,
    gradient_checkpointing=False,
    deepspeed="ds_config.json",
    dataset_text_field="text"
)

trainer = SFTTrainer(
    "THUDM/glm-4-voice-9b",
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config
)

trainer.train()
