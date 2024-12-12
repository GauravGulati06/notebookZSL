from boolq_dataset import BoolQDataset
from gsm8k_dataset import GSM8KDataset
import random
from collections import defaultdict

class DatasetWrapper:

    def __init__(self, dataset_tag, base_dir, split):
        if dataset_tag == "boolq":
            self.dataset = BoolQDataset(
                base_dir=base_dir,
                split=split
            )

        elif dataset_tag == "gsm8k":
            self.dataset = GSM8KDataset(
                base_dir=base_dir,
                split=split
            )
            
        else:
            raise ValueError(f"Unsupported dataset_tag: {dataset_tag}")

    def __len__(self):
        return len(self.dataset)

    def get_dataset(self):
        return self.dataset
    
    def get_random_samples(self, num_samples, seed=None):
        """Get a list of random samples from the dataset."""
        if seed is not None:
            random.seed(seed)
            
        num_samples = min(num_samples, len(self.dataset))
        indices = random.sample(range(len(self.dataset)), num_samples)
        return [self.dataset[i] for i in indices]
