import torch
import os
from typing import List
from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from speech_tokenizer.flow_inference import AudioDecoder


class GLM4CodecEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        tokenizer_path = "THUDM/glm-4-voice-tokenizer"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    def forward(self, audio_path: List[str]) -> torch.Tensor:
        """
        Input: audio_path: list of paths to the audio files
        Output: audio_tokens: (B, T)
        """
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, audio_path
        )  # 12.5 TPS
        audio_tokens = torch.tensor(audio_tokens)
        return audio_tokens


class GLM4CodecDecoder(torch.nn.Module):
    def __init__(self, flow_path):
        """
        flow_path: path to the cloned glm-4-voice-decoder repo
        """
        super().__init__()
        flow_config = os.path.join(flow_path, "config.yaml")
        flow_checkpoint = os.path.join(flow_path, "flow.pt")
        hift_checkpoint = os.path.join(flow_path, "hift.pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                hift_ckpt_path=hift_checkpoint,
                device=device)

    def forward(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Input: audio_tokens: (B, T)
        Output: tts_speech: (B, T)
        """
        tts_speech = self.audio_decoder.offline_inference(audio_tokens)
        return tts_speech
