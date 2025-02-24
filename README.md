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
