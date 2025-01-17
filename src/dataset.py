from typing import List
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            padding_side='right',
            max_length=max_length,
            return_tensors='pt'
        )
        # Create input_ids and labels
        self.input_ids = self.encodings['input_ids']
        self.attention_mask = self.encodings['attention_mask']
        # For causal language modeling, labels are the same as input_ids
        self.labels = self.input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

