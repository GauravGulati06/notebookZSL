import os
import csv
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
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for idx, row in enumerate(csv_reader):  # Add idx while reading
                question = row.get('question', '')
                answer = row.get('answer', '')
                data.append({
                    'idx': idx,         # Add idx directly in the data
                    'question': question,
                    'answer': answer
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
