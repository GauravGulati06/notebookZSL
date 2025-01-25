# Zero-Shot Learning Capabilities of LLMs

This research project evaluates the zero-shot learning capabilities of three large language models (LLMs) on the **Google BoolQ dataset**. The models tested are:
- **Qwen2-1.5B**
- **Llama3-2-3B**
- **Gemma-2-2B-IT**

The project focuses on assessing the performance of these models in zero-shot question-answering tasks and documents their accuracies.

---

## 📊 Results

The accuracies achieved by the models on the Google BoolQ dataset are as follows:
- **Qwen2-1.5B**: 79%
- **Llama3-2-3B**: 83%
- **Gemma-2-2B-IT**: 64%

---

## 🛠️ Setup

### Prerequisites
- Python 3.8 or higher
- Hugging Face Transformers library
- Datasets library (for loading the BoolQ dataset)
- PyTorch or TensorFlow (depending on the model)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/zero-shot-llms.git
   cd zero-shot-llms
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

🤝 Contributing
Contributions are welcome! If you'd like to contribute, please:

Fork the repository.

Create a new branch for your feature or bugfix.

Submit a pull request with a detailed description of your changes.

📜 License
This project is licensed under the GPL-3.0 License. See the LICENSE file for details.

🙏 Acknowledgments
Hugging Face for providing the Transformers library and pre-trained models.

Google for the BoolQ dataset.

Kaggle for hosting the dataset and providing a platform for data science research.
