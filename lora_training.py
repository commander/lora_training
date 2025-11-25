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
instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint-1/checkpoint-9345/", dtype=torch.bfloat16)

# Evaluate the model quality (human evaluation)
index = 200
dialogue = dataset["test"][index]["dialogue"]
human_baseline_summary = dataset["test"][index]["summary"]

prompt = f"""
Summarize the following conversation.

{dialogue}
Summary:
"""

input_ids = tokenizer(prompt, return_tensors='pt').input_ids

original_model_outputs = original_model.generate(
    input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
)
original_model_text_output = tokenizer.decode(
    original_model_outputs[0],
    skip_special_tokens=True
)

instruct_model_outputs = instruct_model.generate(
    input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
)
instruct_model_text_output = tokenizer.decode(
    instruct_model_outputs[0],
    skip_special_tokens=True
)

print(dash_line)
print(f'INPUT PROMPT: \n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY: \n{human_baseline_summary}\n')
print(dash_line)
print(f'ORIGINAL MODEL GENERATED SUMMARY - ZERO SHOT: \n{original_model_text_output}')
print(dash_line)
print(f'INSTRUCT MODEL GENERATED SUMMARY - ZERO SHOT: \n{instruct_model_text_output}')
print(dash_line)


# Evaluate the model qualitatively - with ROUGE Metric
rouge = evaluate.load("rouge")

# Generate the outputs for the sample of the test dataset 
# (only 10 dialogues and summaries to save time)
# and save the results
dialogues = dataset["test"][0:10]["dialogue"]
human_baseline_summaries = dataset["test"][0:10]["summary"]
original_model_summaries = []
instruct_model_summaries = []

for _, dialogue in enumerate(dialogues):
    prompt = f"""
    Summarize the following conversation.

    {dialogue}
    Summary:
    """
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    original_model_outputs = original_model.generate(
        input_ids,
        generation_config=GenerationConfig(max_new_tokens=200)
    )
    original_model_text_output = tokenizer.decode(
        original_model_outputs[0],
        skip_special_tokens=True
    )
    original_model_summaries.append(original_model_text_output)

    instruct_model_outputs = instruct_model.generate(
        input_ids,
        generation_config=GenerationConfig(max_new_tokens=200)
    )
    instruct_model_text_output = tokenizer.decode(
        instruct_model_outputs[0],
        skip_special_tokens=True
    )
    instruct_model_summaries.append(instruct_model_text_output) 

zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries))
df = pd.DataFrame(zipped_summaries, columns=["human_baseline_summary", "original_model_summary", "instruct_model_summary"])

print(df)

# Evaluate the model quantitatively - with ROUGE Metric 
original_model_result = rouge.compute(
    references=human_baseline_summaries[0:len(original_model_summaries)],
    predictions=original_model_summaries,
    use_stemmer=True,
    use_aggregator=True,
)

instruct_model_result = rouge.compute(
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    predictions=instruct_model_summaries,
    use_stemmer=True,
    use_aggregator=True,
)   
print('ORIGINAL MODEL:')
print(original_model_result)
print('INSTRUCT MODEL:')
print(instruct_model_result)

# the file data/dialogue_summary_result.csv contains a pre-populated list of all model results
# which we can use to evaluate on a larger section of data.
results = pd.read_csv('data/dialogue_summary_result.csv')

human_baseline_summaries = results['human_baseline_summaries'].values
original_model_summaries = results['original_model_summaries'].values
instruct_model_summaries = results['instruct_model_summaries'].values    

original_model_result = rouge.compute(
    references=human_baseline_summaries[0:len(original_model_summaries)],
    predictions=original_model_summaries,
    use_stemmer=True,
    use_aggregator=True,
)

instruct_model_result = rouge.compute(
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    predictions=instruct_model_summaries,
    use_stemmer=True,
    use_aggregator=True,
)   

print('Results for all dialogues')
print(dash_line)
print('ORIGINAL MODEL:')
print(original_model_result)
print('INSTRUCT MODEL:')
print(instruct_model_result)

print('Absolute pecentage improvement of Instruct model over Human Baseline:')
improvement = np.array(list(instruct_model_result.values())) - np.array(list(original_model_result.values()))
for key, value in zip(instruct_model_result.keys(), improvement):
    print(f'{key}: {value*100:.2f}')

# PEFT - Parameter Efficient Fine-Tuning
# PEFT is a form of instruction fine-tuning that uses a smaller number of parameters to fine-tune a model  
# PEFT is a generic term that includes LoRA (Low-Rank Adaptation) and Prompt Tuning.

# Setup PEFT/LoRA model for fine-tuning
# With LoRA you freeze the underlying model parameters and only train the adapter.
from peft import LoraConfig, get_peft_model, TaskType
def train_lora():
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, # FLAN-T5
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q", "v"]
    )
    
    # add lora adapter layers/parameters to the original LLM to be trained
    peft_model = get_peft_model(original_model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))
    
    # Train PEFT Adapter
    output_dir = f"./trained_models/peft-dialogue-summary-training-{str(int(time.time()))}"
    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3, # Higher learning rate than full fine-tuning
        num_train_epochs=3,
        logging_dir="logs",
        logging_steps=1,
        # max_steps=1,
        remove_unused_columns=False,
        # other parameters to have 
        weight_decay=0.01,
        # eval_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # report_to="none",
        per_device_train_batch_size=4,  # Reduce batch size to save memory
        gradient_accumulation_steps=1,   # Accumulate gradients to simulate larger batch
        dataloader_pin_memory=False,      # Disable pin_memory for MPS compatibility
        max_grad_norm=1.0,  # CLIP gradients to prevent explosion (this will help!)
        eval_strategy='steps',  # Evaluate every N steps (older transformers uses eval_strategy)
        eval_steps=500,  # Evaluate every 30 steps
        save_strategy='steps',  # Save checkpoints every N steps
        save_steps=1000,  # Save every 60 steps
        save_total_limit=2,  # Only keep 2 most recent checkpoints to save disk space
        load_best_model_at_end=True,  # Load the best checkpoint at the end
        metric_for_best_model='loss',  # Use validation loss to determine best model
    )

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    peft_trainer.train()
    peft_model_path = "./trained_models/peft-dialogue-summary-checkpoint-local"

    # save model and tokenizer
    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)

# train_lora()

# now the lora is trained, prepare this model by adding the adapter to the original FLAN-T5 model.
# `is_trainable` parameter is set to false because the plan is only to perform inference with this PEFT model.
# First load the base model
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

from peft import PeftModel, PeftConfig
peft_model = PeftModel.from_pretrained(
    peft_model_base,
    "./trained_models/trained-lora-adapter/",
    dtype=torch.bfloat16,
    is_trainable=False
)

# the number of trainable parameters will be 0 due to is_trainable=False setting
print(print_number_of_trainable_model_parameters(peft_model))

# Evaluate the Model Qualitatively (Human Evaluation)
index = 200
dialogue = dataset['test'][index]['dialogue']
baseline_human_summary = dataset['test'][index]['summary']

print(f"Dialogue: {dialogue}")
print(f"Baseline Human Summary: {baseline_human_summary}")

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

original_model_outputs = original_model.generate(
    input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
) 
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

instruct_model_outputs = instruct_model.generate(
    input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
) 
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
) 
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

print(dash_line)
print("Evaluate the Model Qualitatively (Human Evaluation)")
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
print(f'ORIGINAL MODEL SUMMARY:\n{original_model_text_output}')
print(f'INSTRUCT MODEL SUMMARY:\n{instruct_model_text_output}')
print(f'PEFT MODEL SUMMARY:\n{peft_model_text_output}')

# Evaluate the Model Quantitatively (with ROUGE Metric)

dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []
instruct_model_summaries = []
peft_model_summaries = []

for idx, dialogue in enumerate(dialogues):
    prompt = f"""
    Summarize the following conversation.

    {dialogue}

    Summary: """
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    human_baseline_text_output = human_baseline_summaries[idx]
    original_model_outputs = original_model.generate(
        input_ids,
        generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
    ) 
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    instruct_model_outputs = instruct_model.generate(
        input_ids,
        generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
    ) 
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
    peft_model_outputs = peft_model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
    ) 
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
    original_model_summaries.append(original_model_text_output)
    instruct_model_summaries.append(instruct_model_text_output)
    peft_model_summaries.append(peft_model_text_output) 

zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries, peft_model_summaries))

df = pd.DataFrame(zipped_summaries, columns=['human_baseline_summary', 'original_model_summary', 'instruct_model_summary', 'peft_model_summary'])

print(df)

rouge = evaluate.load("rouge")

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print(dash_line)
print("Evaluate the Model Quantitatively (with ROUGE Metric)")
print(dash_line)
print("Original Model Results:")
print(original_model_results)
print("Instruct Model Results:")
print(instruct_model_results)
print("PEFT Model Results:")
print(peft_model_results)

# Now calculate the  ROUGE score on the full dataset with PEFT model.
# and check the performance compared to other models.
human_baseline_summaries = results['human_baseline_summaries'].values
original_model_summaries = results['original_model_summaries'].values
instruct_model_summaries = results['instruct_model_summaries'].values
peft_model_summaries = results['peft_model_summaries'].values

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print(dash_line)
print("Evaluate the Model Quantitatively (with ROUGE Metric)")
print(dash_line)
print("Original Model Results:")
print(original_model_results)
print("Instruct Model Results:")
print(instruct_model_results)
print("PEFT Model Results:")
print(peft_model_results)

# Calculate the improvement of PEFT over the original model
print('Absolute percentage improvement of PEFT model over HUMAN Baseline')
improvement = np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values()))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')

# Calculate the improvement of PEFT over full fine-tuned model
print('Absolute percentage improvement of PEFT model over full fine-tuned model')
improvement = np.array(list(peft_model_results.values())) - np.array(list(instruct_model_results.values()))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')

