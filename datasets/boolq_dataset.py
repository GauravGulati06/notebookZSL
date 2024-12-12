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
        data = []
        with open(self.data_path, 'r') as f:
            for idx, line in enumerate(f):  # Add idx while reading
                sample = json.loads(line)
                question = sample.get('question', None)
                passage = sample.get('passage', None)
                label = sample.get('answer', None)
                data.append({
                    'idx': idx,        # Add idx directly in the data
                    'question': question,
                    'passage': passage,
                    'label': label
                })
        return data

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
        sample = self.data[idx]
        return sample  # No need to extract 'idx' here since it's already part of the data
