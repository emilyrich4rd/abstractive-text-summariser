# abstractive-text-summariser
Abstractive text summariser made with the T5 small model and Hugging Face Transformers. Both a demo version and a full GPU-training version are included.

## About the project
My project is an abstractive text summariser with a simple text-based user interface. There are two versions; one is a lightweight demo version running on fewer resources, while the other requires a GPU to run but produces a more accurate abstractive summary. I undertook this as a learning project, to implement a transformer model that uses encoder-decoder architecture in practice. I also wanted to learn about the natural language processing pipeline, including key steps such as tokenisation. The basic structure of the project consists of tokenisation, training and evaluation, and passing the text from user input into the model.

## Features and Versions
In both versions, the T5 language model has been trained on a dataset from Hugging Face consisting of CNN and Daily Mail articles and their corresponding summaries. In terms of the differences, the demo uses T5-small and only a subset of the training dataset is used, while the full GPU-training version uses T5-base and all of the available data in the dataset. Batch size is 16 for the demo but 8 for the full GPU-training version, as smaller batch size creates a more generalised model. The number of training epochs is 1 for the demo, but 3 for the full GPU-training version.

The full GPU-training version should therefore produce a more accurate, abstractive summary than the demo version. Due to resource limitations, I have not been able to fully test it (unlike with the demo version) but both versions are very similar.

## Getting Started
### Prequisites
- Python 3.9+
- If running the full GPU-training version, a CUDA-compatible GPU is  required
- Python package requirements are outlined in requirements.txt

### Installation
1. Clone the repo:
````
  git clone https://github.com/github_username/repo_name.git
````
2. Install dependencies
````
pip install -r requirements.txt
````
3. Run either version of the text summariser, and input text to be summarised when instructed to do so

## Usage
On the first run, the model needs to be trained before the summary tool can be used, so metrics from training will be displayed such as validation loss. On subsequent runs, the model is saved, so the tool can be instantly used to provide a text-summary.
