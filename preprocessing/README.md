# How we preprocess the data

1. Download the `rpg_overlap_30_35` split from our nas.

2. Run `preprocess.py` to convert the timestamp csv files to huggingface dataset format.

3. Run `build_dataset.py` to build the final dataset for training:
    - Extract GLM-4 codec tokens from the audio files
    - Remove overlap messages
    - Clean the text

The final dataset is [here](https://huggingface.co/datasets/anthony-wss/rpg-overlap-30-35-processed)
