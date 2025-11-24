# imports
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

# load dataset and LLM
## we are going to use the DialogSum huggingface dataset. It contains the 10000+ dialogues with the corresponding manually labeled summaries and topics
hugginface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(hugginface_dataset_name)
dataset

# Load the pre-trained FLAN-T5 model and its tokenizer directly from huggingface. 
# Notice that we are using the small version of FLAN-t5. 
# Setting `torch_dtype=torch.bfloat16` specifies the memory to be used by this model
model_name="google/flan-t5-base"
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get the number of model parameters and find how many are trainable

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model params: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))

# Test the model with the zero shot inferencing. 
# You can see that the model struggles to summarize the dialogue compared to the baseline summary,
# but it does pull out some important information from the text 
# which indicates the model can be fine-tuned to the task at hand.

index = 200
dialogue = dataset["test"][index]["dialogue"]
summary = dataset["test"][index]["summary"]

prompt = f"""
Summarize the following conversation.

{dialogue}
Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(original_model.generate(
    inputs["input_ids"],
    max_new_tokens=200,
)[0], skip_special_tokens=True)

dash_line = "-".join("" for x in range(100))
print(dash_line)
print(f'INPUT PROMPT: \n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY: \n{summary}\n')
print(dash_line)
print(f'MODEL GENERATED SUMMARY - ZERO SHOT: \n{output}')

# 2. Perform full fine-tuning

# 2.1 Preprocess the Dialog-Summary Dataset
## Each row of the dataset (prompt = dialog, response = summary) is converted to 
## instruction for the LLM in this format
## Training prompt (dialogue):
##      Summarize the following conversation.
##          Chris: This is his part of the conversation.
##          Antje: This is her part of the conversation.
##      Summary:
##
## Training Response (Summary):
##
##      Both Chris and Antje participated in the conversation.
##
## Then, preprocess the prompt-response dataset into tokens and pull out their input_ids (1 per token).

def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True).input_ids
    example['labels'] = tokenizer(example['summary'], padding="max_length", truncation=True).input_ids

    return example

# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

# # To save time in the lab, subsample the dataset to 1% of the original size
# tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)

# Check the Shapes of all three parts of the dataset
print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)

# 2.2 - Fine-tune the model with the preprocessed dataset
## we will utilize built-in Huggingface `Trainer` class. 
## Pass the preprocessed dataset with reference to the original model. 

output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

def train_model(output_dir, tokenized_datasets, model):
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,  # Reduced to help with gradient stability
        num_train_epochs=3,  # Reduced for learning/experimentation
        weight_decay=0.01,
        logging_steps=20,  # Log frequently to see progress
        # max_steps=300,  # Limit total steps for quick experimentation
        per_device_train_batch_size=4,  # Reduce batch size to save memory
        gradient_accumulation_steps=1,   # Accumulate gradients to simulate larger batch
        dataloader_pin_memory=False,      # Disable pin_memory for MPS compatibility
        max_grad_norm=1.0,  # CLIP gradients to prevent explosion (this will help!)
        eval_strategy='steps',  # Evaluate every N steps (older transformers uses eval_strategy)
        eval_steps=50,  # Evaluate every 30 steps
        save_strategy='steps',  # Save checkpoints every N steps
        save_steps=200,  # Save every 60 steps
        save_total_limit=2,  # Only keep 2 most recent checkpoints to save disk space
        load_best_model_at_end=True,  # Load the best checkpoint at the end
        metric_for_best_model='loss',  # Use validation loss to determine best model
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation']
    )

    trainer.train()

# train_model(output_dir, tokenized_datasets, original_model)

# Create an instance of AutoModelForSeq2SeqLM class for the instruct model.
model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint-1", torch_dtype=torch.bfloat16)