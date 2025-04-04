from datasets import load_dataset
import datetime

import os
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser
import json
import wandb
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run_name = f"glm-lora-ft-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"./experiments/{run_name}"
os.makedirs(output_dir, exist_ok=True)

wandb.init(project="glm-voice-finetune", name=run_name, config=config)

with open(os.path.join(output_dir, "config.yaml"), "w") as f:
    yaml.dump(config, f)

MAX_LENGTH = config["max_length"]

train_dataset = load_dataset("anthony-wss/rpg-overlap-30-35-processed", split="train")
eval_dataset = load_dataset("anthony-wss/rpg-overlap-30-35-processed", split="test")
print("Dataset loaded")

peft_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    target_modules=config["lora_trainable"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM"
)

sft_config = SFTConfig(
    output_dir=output_dir,
    max_seq_length=MAX_LENGTH,
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["train_bsz"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    per_device_eval_batch_size=config["eval_bsz"],
    warmup_steps=config["warmup_steps"],
    logging_strategy="steps",
    logging_steps=config["logging_steps"],
    save_strategy="steps",
    save_steps=config["save_steps"],
    num_train_epochs=config["num_epochs"],
    bf16=True,
    gradient_checkpointing=False,
    deepspeed="ds_config.json",
    dataset_text_field="text",
    report_to="wandb"
)

model = AutoModel.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config
)

trainer.train()
