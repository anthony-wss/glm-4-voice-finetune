import torch
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

def FineTunedGLM4Voice(model_path: str):
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=None,
        device_map="auto"
    )
    return model

def GLM4VoiceTokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def prepare_model_for_lora(model):
    # Configure LoRA specifically for ChatGLM architecture
    lora_config = LoraConfig(
        r=8,
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
    # Create LoRA model on CPU
    device = model.device
    model = model.to("cpu")

    # Prepare model for k-bit training if needed
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    model = model.to(device)
    
    return model
