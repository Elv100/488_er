import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
# import bitsandbytes as bnb

# Quantize the Base Model
def load_quantized_model(model_name, use_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_quant_type="nf4", use_nested_quant=False):
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

# Load Training Dataset
def load_training_dataset(dataset_path):
    return load_from_disk(dataset_path)

# Attach an Adapter Layer
def attach_adapter_layer(model):
    modules = ["q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj"]
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules
    )
    return peft_config

# Find all Linear Layers
# def find_all_linear_names(model):
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, bnb.nn.Linear4bit):
#             names = name.split(".")
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])

#     if "lm_head" in lora_module_names:  # needed for 16-bit
#         lora_module_names.remove("lm_head")
#     return list(lora_module_names)

# Train the Model
def train_model(model, dataset, peft_config, training_arguments):
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]), 
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[2] for f in data])}
    )
    trainer.train()
    return trainer

# Main Execution
def main():
    # Model and dataset names
    model_name = "meta-llama/Llama-2-7b-hf"
    dataset_path ='s3://sagemaker-us-east-2-851725296592/processed/optimProxCause/train'
    
    # Load quantized model and tokenizer
    model, tokenizer = load_quantized_model(model_name)

    # Load dataset
    dataset = load_training_dataset(dataset_path)

    # Attach adapter layer
    peft_config = attach_adapter_layer(model)

    # Set training parameters
    output_dir = "/tmp"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        # bf16=args.bf16,  # Use BF16 if available
        learning_rate=2e-4,
        num_train_epochs=3,
        # gradient_checkpointing=args.gradient_checkpointing,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
    )

    # Train model
    trainer = train_model(model, dataset, peft_config, training_args)

    # Save trained model
    # Upload model to Amazon S3
    save_path = 's3://sagemaker-us-east-2-851725296592/opt/ml/model'
    trainer.model.save_pretrained(save_path)

    
if __name__ == "__main__":
    main()
