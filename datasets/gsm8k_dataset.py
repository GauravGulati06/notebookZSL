import os
import json
import pandas as pd
from torch.utils.data import Dataset

class GSM8KDataset(Dataset):
    def __init__(self, base_dir, split):
        """
        Args:
            base_dir (str): Path to the base folder containing dataset splits.
            split (str): Dataset split to use ('train', 'test').
        """
        self.data_path = os.path.join(base_dir, f"main_{split}.csv")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset split file not found: {self.data_path}")
        self.data = self._load_data()

    def _load_data(self):
        """Loads data from the CSV file."""
        return pd.read_csv(self.data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._get_item(i) for i in range(*idx.indices(len(self)))]

        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.data):
                raise IndexError(f"Index {idx} is out of range.")

        return self._get_item(idx)

    def _get_item(self, idx):
        sample = self.data.iloc[idx]
        question = sample['question'] if 'question' in sample else None
        answer = sample['answer'] if 'answer' in sample else None

        return {
            'question': question,
            'answer': answer
        }