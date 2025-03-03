# GLM-4-Voice Fine-tuning

This project implements fine-tuning of the GLM-4-Voice model.

## Install

```bash
# Make sure git lfs is installed
git clone https://huggingface.co/THUDM/glm-4-voice-decoder

# This can take ~5min (because of matcha-tts)
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python train.py
```

## Train with multi-nodes on TWCC

```bash
sbatch train-multinodes.sbatch
```

## Inference

GLM-4-Voice supports streaming inference. However, currently `inference.py` only supports offline generation and is designed for model evaluation. For realtime demo, please refer to [gradio-demo](https://github.com/THUDM/GLM-4-Voice?tab=readme-ov-file#launch-web-demo)

It would required ~20GB VRAM to run the model in `float16`.

```bash
python inference.py --input_audio <path/to/input/audio> --output_audio <path/to/save/output/audio>
```
