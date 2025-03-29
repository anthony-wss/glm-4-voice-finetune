""" Some code is borrowed from https://github.com/THUDM/GLM-4-Voice/blob/main/web_demo.py """

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from argparse import ArgumentParser
from src.vocoder import GLM4CodecEncoder, GLM4CodecDecoder
import torch
import soundfile as sf
from peft import AutoPeftModelForCausalLM

def main(input_audio: str, output_audio: str):
    # Load all the models
    glm_model = AutoModel.from_pretrained(
        "THUDM/glm-4-voice-9b",
        trust_remote_code=True,
        quantization_config=None,
        device_map={"": 0}
    )

    # For peft model
    # glm_model = AutoPeftModelForCausalLM.from_pretrained(
    #     "output/checkpoint-1000",
    #     device_map="cuda",
    #     trust_remote_code=True
    # )

    glm_tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
    glm_speech_encoder = GLM4CodecEncoder()
    glm_speech_decoder = GLM4CodecDecoder("./glm-4-voice-decoder")

    # Load the user input audio
    audio_tokens = glm_speech_encoder([input_audio])
    audio_tokens = audio_tokens[0]
    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
    user_input = audio_tokens

    # Build the prompt
    system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
    inputs = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

    # Run model inference
    with torch.no_grad():
        inputs = glm_tokenizer(inputs, return_tensors="pt").to(glm_model.device)
        outputs = glm_model.generate(**inputs, temperature=0.2, top_p=0.8, max_new_tokens=2000)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    # Decode the output
    def is_audio_token(token_id):
        """ source: https://github.com/THUDM/GLM-4-Voice/blob/eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a/web_demo.py#L162 """
        return token_id >= glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')

    audio_tokens, text_tokens = [], []
    for token in generated_tokens:
        if is_audio_token(token):
            audio_tokens.append(token)
        else:
            text_tokens.append(token)
    print("Model audio output:", glm_tokenizer.decode(audio_tokens))
    print("Model text output:", glm_tokenizer.decode(text_tokens, skip_special_tokens=True))

    # Synthesize the audio & save
    audio_offset = glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
    tts_speech = glm_speech_decoder(torch.tensor([[atok - audio_offset for atok in audio_tokens]]))
    sf.write(output_audio, tts_speech.squeeze(), 22050)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_audio", type=str, required=True)
    parser.add_argument("--output_audio", type=str, default="model_output.wav")
    args = parser.parse_args()
    main(args.input_audio, args.output_audio)
