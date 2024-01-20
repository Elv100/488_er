# Inspired by, but substantially modified from HuggingFace's Sagemaker tutorial
# https://github.com/huggingface/notebooks/blob/main/sagemaker/28_train_llms_with_qlora/sagemaker-notebook.ipynb
import os
from transformers import AutoTokenizer
from itertools import chain
from functools import partial
import torch
from random import randint, randrange
from datasets import Dataset
from sagemaker_setup import sess

# Path to your dataset folder
dataset_folder_path = './Optimized_Cases'

# Read dataset
def read_dataset(folder_path):
    dataset = []
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                background, verdict = content.split("Verdict:")
                dataset.append({'background': background, 'verdict': verdict})
    return dataset

raw_dataset = read_dataset(dataset_folder_path)

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({'data': raw_dataset})

# Model and Tokenizer
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Formatting function
def format_case(sample):
    background = f"### Background\n{sample['data']['background']}"
    verdict = f"### Verdict\n{sample['data']['verdict']}"
    return {'text': background + "\n\n" + verdict + tokenizer.eos_token}

# Apply formatting
dataset = dataset.map(format_case)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Print a sample to verify
print(dataset[0]['text'])

# Print total number of samples
print(f"Total number of samples: {len(tokenized_dataset)}")

# Save dataset to disk and upload to S3
training_input_path = f's3://{sess.default_bucket()}/processed/optimProxCause/train'
tokenized_dataset.save_to_disk(training_input_path)

print("uploaded data to:")
print(f"training dataset to: {training_input_path}")