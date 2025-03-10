"""
Convert the timestamp csv files to huggingface dataset format.
"""

import csv
from datasets import Dataset
import os
from tqdm import tqdm

def parse_timestamp(timestamp_file):
    conversation = []
    with open(timestamp_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        next(spamreader)
        cur_speaker = None
        cur_text = ""
        cur_start = None
        cur_end = None
        for row in spamreader:
            row = row[0].split(",")
            if cur_speaker is None:
                cur_speaker = row[0]
                cur_start = row[1][:-1]
                cur_end = row[2][:-1]
            else:
                if cur_speaker == row[0]:
                    cur_text += row[3]
                    cur_end = row[2][:-1]
                else:
                    conversation.append({
                        "speaker": int(cur_speaker),
                        "text": cur_text,
                        "start": float(cur_start),
                        "end": float(cur_end)
                    })
                    cur_speaker = row[0]
                    cur_text = row[3]
                    cur_start = row[1][:-1]
                    cur_end = row[2][:-1]
        conversation.append({
            "speaker": int(cur_speaker),
            "text": cur_text,
            "start": float(cur_start),
            "end": float(cur_end)
        })
    return conversation


dataset = []

for timestamp_file in tqdm(os.listdir("/work/u3937558/tp2/_rpg_overlap_30_35/timestamp")):
    conversation = parse_timestamp(os.path.join("/work/u3937558/tp2/_rpg_overlap_30_35/timestamp", timestamp_file))
    dataset.append({
        "conversation": conversation,
        "audio_file": timestamp_file.replace(".csv", ".wav"),
        "timestamp_file": timestamp_file
    })

dataset = Dataset.from_list(dataset)
dataset.save_to_disk("./hf_rpg_overlap_30_35")
