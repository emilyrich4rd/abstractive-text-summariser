from datasets import load_dataset, load_from_disk, DatasetDict
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import DataLoader

if torch.cuda.is_available(): # checks for available GPU
  device = torch.device("cuda") # uses GPU
else:
  device = torch.device("cpu") # uses CPU

cache_directory = "./hugging_face_dataset" # specifies name of folder where dataset will be stored

if os.path.exists(os.path.join(cache_directory, "cnn_dailymail")): # loads Hugging Face dataset from local cache folder if already present
  dataset = load_from_disk(os.path.join(cache_directory, "cnn_dailymail"))
else: # downloads Hugging Face dataset if the local cache folder does not exist
  dataset = load_dataset("cnn_dailymail", "3.0.0") # specifies the dataset to be downloaded
  os.makedirs(cache_directory, exist_ok=True) # makes a directory to store the dataset, if it does not already exist
  dataset.save_to_disk(os.path.join(cache_directory, "cnn_dailymail")) # saves dataset to the cache folder

# splits data into training, validation and testing
# for demo purposes a subset of the data is used
train_set = dataset["train"].select(range(20000))
val_set   = dataset["validation"].select(range(5000))
test_set  = dataset["test"].select(range(5000))

# merges the 3 datasets into a single object
dataset = DatasetDict({
    "train": train_set,
    "validation": val_set,
    "test": test_set
})

# name of folder where the model will be saved, or is saved (if the script has been run before)
model_path = "./saved_model"

def initialise_model():
  if os.path.exists(model_path): # if the script has been run before, the model has been saved locally and can be loaded from local memory
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
  else:
    print("Training new model")
    # imports Hugging Face's t5-small model, the lightweight version of Google's T5 model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    # imports tokenizer to convert text into tokens to be processed by the model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def tokenise_data(my_set):
      # adds the prefix "summarise" to each input string to indicate the task is abstractive summarisation
      input_list = []
      for article in my_set["article"]:
        input_list.append("summarize:" + article)

      # converts input strings to tokens
      inputs = tokenizer(
          input_list, # specifies input source
          max_length = 512, # max. number of tokens the model can take as input
          truncation = True, # inputs that exceed max. number of tokens are truncated
          padding = "max_length" # pads smaller inputs so all are 512 tokens
      )

      # converts training outputs (summaries) to tokens
      labels = tokenizer(
          my_set["highlights"], # refers to the summaries of the articles
          max_length = 150, # max. number of tokens the summary will be
          truncation = True, # summaries that exceed max. number of tokens are truncated
          padding = "max_length" # pads smaller summaries so all are 150 tokens
      )

      # goes through the 2d array of summary labels and ensures any padding tokens will not be used by the model to calculate loss
      # -100 is the ID for tokens to be ignored
      for label in labels["input_ids"]:
        for token_id in range(len(label)):
          if label[token_id] == tokenizer.pad_token_id:
            label[token_id] = -100

      # merges information into a dictionary to be returned as a single object
      return {
          "input_ids": inputs["input_ids"],
          "attention_mask": inputs["attention_mask"], # indicates padded tokens so they can be ignored by the model
          "labels": labels["input_ids"]
      }

    # applies tokenise_data function to each partition (train, val, test) as entire dictionary cannot be passed into the function
    # batched = True ensures data is passed as batches, not individually, for efficiency
    tokenised_datasets = dataset.map(tokenise_data, batched = True)

    # splits tokenised_datasets into datasets for training, validation and testing
    tokenised_training_data = tokenised_datasets["train"]
    tokenised_validation_data  = tokenised_datasets["validation"]
    tokenised_testing_data  = tokenised_datasets["test"]

    # specifies training parameters
    training_parameters = TrainingArguments(
      output_dir = "./results", # specifies location to save outputs of the model
      report_to = "none", # prevents external metric reports
      num_train_epochs = 1, # number of passes over the dataset during training
      per_device_train_batch_size = 16, #  training batch size per CPU/GPU
      per_device_eval_batch_size = 16, # evaluation batch size per CPU/GPU
      weight_decay = 0.01, # regularisation to prevent overfitting
      learning_rate = 0.0001, # learning rate for AdamW optimiser
      eval_strategy = "epoch", # visible metrics for evaluation shown at end of each epoch
      save_strategy = "epoch", # saves model after every epoch
      load_best_model_at_end = True # loads model with the least eval loss
    )

    # sets up training with the relevant model, parameters and datasets
    trainer = Trainer(
      model = model,
      args = training_parameters,
      train_dataset = tokenised_training_data,
      eval_dataset = tokenised_validation_data
    )

    # training occurs and results are displayed
    trainer.train()
    test_results = trainer.evaluate(eval_dataset = tokenised_testing_data)
    print(test_results)

    # model is saved locally in the model
    if not os.path.exists(model_path):
      os.makedirs(model_path)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

  return model, tokenizer

# sets up model and tokenizer
model, tokenizer = initialise_model()

# wraps all other functions into one to allow the model to run on the actual user input
def summarise_text(input_text):
  inputs = tokenizer( # input text is converted into tokens
        "summarize: " + input_text, # adds the prefix "summarise" to each input string to indicate the task is abstractive summarisation
        return_tensors = "pt",  # returns as pytorch tensors
        max_length = 512, # max. number of tokens that can be taken as input
        truncation = True, # inputs that exceed max. number of tokens are truncated
        padding = "max_length" # pads smaller inputs so all are 512 tokens
    ).to(device) # ensures tensors are on the correct device, CPU or GPU based on specific machine running the script

  summary_ids = model.generate( # summary generation occurs
      inputs["input_ids"],
      max_length = 150, # max. number of tokens the summary will be
      min_length = 30, # min. number of tokens the summary will be
      num_beams = 4 # ensures high-quality summaries by exploring several possible sequences
  )

  # converts tokens to a text summary without any special tokens
  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return summary

# gets input text from user and passes into summarisation function
article = input("Enter an article to be summarised")
summary = summarise_text(article)
print("Summary:", summary)

