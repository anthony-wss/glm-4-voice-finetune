from src.vocoder import GLM4CodecEncoder, GLM4CodecDecoder
import os
import soundfile as sf

def test_vocoder():
    encoder = GLM4CodecEncoder()
    decoder = GLM4CodecDecoder(os.path.join(os.path.dirname(__file__), "../glm-4-voice-decoder"))

    audio_path = os.path.join(os.path.dirname(__file__), "../input-sample-0.wav")
    audio_tokens = encoder([audio_path])
    tts_speech = decoder(audio_tokens)
    assert tts_speech.shape[0] == 1, f"tts_speech.shape = {tts_speech.shape}"
    assert tts_speech.shape[1] > 0, f"tts_speech.shape = {tts_speech.shape}"
    sf.write(os.path.join(os.path.dirname(__file__), "../output-sample-0.wav"), tts_speech.squeeze(), 24000)
