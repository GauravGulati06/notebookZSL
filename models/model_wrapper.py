import torch
from transformers import pipeline

models = {
    "qwen2.5_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama3.2_3b": "meta-llama/Llama-3.2-3B-Instruct",
}


class ModelWrapper:
    
    def __init__(self, model_name):

        # Hyperparams
        self.max_new_tokens = 20
        self.temperature = 0.001

        self.model_name = model_name
        self.pipe = pipeline(
            task="text-generation",
            model=model_name,
            device_map="auto"
        )
    
    def generate(self, messages):
        """
        messages format:
        [
            {"role": "user", "content": "Who are you?"},
            ...
        ]
        """
        output = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )

        return output
