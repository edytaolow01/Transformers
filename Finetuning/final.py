import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
import evaluate
import numpy as np

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Set environment variable for CUDA launch blocking for better error debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Choose GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Using device: {device}")

# Load dataset
raw_dataset = load_dataset("WiktorS/polish-news", split="train")
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

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_billsum_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
)

# Evaluation metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

# Initialize the model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Test model on new data (can be added as needed)