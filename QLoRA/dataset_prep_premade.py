from datasets import load_dataset
from random import randrange
from transformers import AutoTokenizer
from random import randint
from itertools import chain
from functools import partial
from sagemaker_setup import sess

# Load dataset from Datasets Hub
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

print(f"dataset size: {len(dataset)}")
print(dataset[randrange(len(dataset))]) # dataset size: 15011

model_id = "meta-llama/Llama-2-7b-hf" # sharded weights
# model_id = "mistralai/Mistral-7B-Instruct-v0.2" # sharded weights
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt

# print(format_dolly(dataset[randrange(len(dataset))]))

# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample


# apply prompt template per sample
dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
# print random sample
print(dataset[randint(0, len(dataset))]["text"])

# empty list to save remainder from batches to use in next batch
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result


# tokenize and chunk dataset
lm_dataset = dataset.map(
    lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
).map(
    partial(chunk, chunk_length=2048),
    batched=True,
)

# Print total number of samples
print(f"Total number of samples: {len(lm_dataset)}")



# save train_dataset to s3
training_input_path = f's3://{sess.default_bucket()}/processed/dolly/train'
lm_dataset.save_to_disk(training_input_path)

print("uploaded data to:")
print(f"training dataset to: {training_input_path}")