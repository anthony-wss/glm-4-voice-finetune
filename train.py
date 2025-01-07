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

def main():
    whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    glm_model = FineTunedGLM4Voice(model_path).model
    glm_tokenizer = GLM4VoiceTokenizer(model_path).tokenizer

    # Prepare `train_texts` consisting of 100 identical audio inputs for debugging
    audio_path = "./input-sample-0.wav"
    audio_tokens = extract_speech_token(
        whisper_model, feature_extractor, [audio_path]
    )[0]  # 12.5 TPS
    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
    user_input = audio_tokens
    system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>streaming_transcription\n"
    train_texts = [prompt] * 100

    train_dataset = TextDataset(train_texts, glm_tokenizer)


    glm_model = prepare_model_for_lora(glm_model)

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


if __name__ == "__main__":
    main()
