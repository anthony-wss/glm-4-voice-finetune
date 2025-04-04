from datasets import load_dataset
import datetime

import os
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_arg("--lora_r", type=int, help="The lora rank")
args = parser.parse()

run_name = f"glm-lora_r{args.lora_r}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = "./experiments/lora_r{args.lora_r}/{run_name}"

wandb.init(project="glm-voice-finetune", name=run_name, config=vars(args))

with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

MAX_LENGTH = 1024

train_dataset = load_dataset("anthony-wss/rpg-overlap-30-35-processed", split="train")
eval_dataset = load_dataset("anthony-wss/rpg-overlap-30-35-processed", split="test")
print("Dataset loaded")

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

sft_config = SFTConfig(
    output_dir="output",
    max_seq_length=MAX_LENGTH,
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    num_train_epochs=3,
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
