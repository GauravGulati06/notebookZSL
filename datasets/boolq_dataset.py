import os
import json
from torch.utils.data import Dataset

class BoolQDataset(Dataset):
    def __init__(self, base_dir, split):
        """
        Args:
            base_dir (str): Path to the base folder containing dataset splits.
            split (str): Dataset split to use ('train', 'test', or 'dev').
        """
        self.data_path = os.path.join(base_dir, f"{split}.jsonl")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset split file not found: {self.data_path}")
        self.data = self._load_data()

    def _load_data(self):
        """Loads data from the JSONL file."""
        with open(self.data_path, 'r') as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.data[i] for i in range(*idx.indices(len(self)))]

        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.data):
                raise IndexError(f"Index {idx} is out of range.")

        sample = self.data[idx]
        question = sample['question']
        passage = sample['passage']
        label = sample['answer'] if 'answer' in sample else None

        return {
            'question': question,
            'passage': passage,
            'label': label
        }