#It do not have a sense for summarzation task

import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Set environment variable for CUDA launch blocking for better error debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Load dataset
raw_dataset = load_dataset("WiktorS/polish-news", split="train").shuffle(seed=42).select(range(100))
raw_dataset = raw_dataset.train_test_split(test_size=0.1)

# Filter out examples with None Types in 'headline' and 'content' columns
def filter_none(example):
    return example['headline'] is not None and example['content'] is not None

raw_dataset = raw_dataset.filter(filter_none)

# Choose model checkpoint
checkpoint = "facebook/bart-large-cnn"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenization function
prefix = "summarize: "

def tokenize_function(example):
    inputs = [prefix + doc for doc in example["content"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=example["headline"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
logging.info(f"Example from tokenized dataset: {tokenized_dataset['train'][0]}")

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

tokenized_dataset = tokenized_dataset.remove_columns(["link", "title", "headline", "content"])


tokenized_dataset.set_format("torch") #UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. batch["labels"] = torch.tensor(batch["labels"],

train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

print("DataLoader created successfully: ", {k: v.shape for k, v in batch.items()})

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print("Training steps:", num_training_steps)

# Choose GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
logging.info(f"Using device: {device}")

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


metric = evaluate.load("rouge")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())        