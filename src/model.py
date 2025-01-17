import torch
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


class FineTunedGLM4Voice:
    def __init__(self, model_path: str):
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=None,
            device_map="cpu"
        )

class GLM4VoiceTokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def prepare_model_for_lora(model):
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
