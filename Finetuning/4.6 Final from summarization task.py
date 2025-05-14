# pip install sentencepiece
#text summarization on polish news dataset

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Set environment variable for CUDA launch blocking for better error debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Load dataset
raw_dataset = load_dataset("WiktorS/polish-news", split="train").shuffle(seed=42).select(range(100)) #features: ['link', 'title', 'headline', 'content'] here the "headline" will be the target and "content" will be the input
raw_dataset = raw_dataset.train_test_split(test_size=0.1)

# Filter out examples with None Types in 'headline' and 'content' columns
def filter_none(example):
    return example['headline'] is not None and example['content'] is not None

raw_dataset = raw_dataset.filter(filter_none)

print(raw_dataset)

# tokenize 

checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Prefix the input with a prompt

prefix = "summarize: " #Some models capable of multiple NLP tasks require prompting for specific tasks.

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["content"][:1500]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["headline"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenize entire dataset

tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# create a batch of examples
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# choose evaluation for summarization

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

# instantiate model

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_billsum_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    eval_accumulation_steps=1,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()