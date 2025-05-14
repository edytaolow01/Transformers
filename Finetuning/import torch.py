import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
import evaluate
import numpy as np

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Choose GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Using device: {device}")

# Load the saved tokenizer and model from the checkpoint directory
checkpoint_dir = "path/to/saved/checkpoint"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir).to(device)

# Load tokenized dataset if already saved, otherwise tokenize and save it
tokenized_dataset_path = "path/to/tokenized/dataset"
try:
    logging.info("Loading tokenized dataset from disk...")
    tokenized_dataset = Dataset.load_from_disk(tokenized_dataset_path)
except FileNotFoundError:
    logging.info("Tokenized dataset not found, processing raw dataset...")
    raw_dataset = load_dataset("WiktorS/polish-news", split="train")
    raw_dataset = raw_dataset.train_test_split(test_size=0.1)
    
    def filter_none(example):
        return example['headline'] is not None and example['content'] is not None

    raw_dataset = raw_dataset.filter(filter_none)

    prefix = "summarize: "

    def tokenize_function(example):
        inputs = [prefix + doc for doc in example["content"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(text_target=example["headline"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    logging.info("Saving tokenized dataset to disk...")
    tokenized_dataset.save_to_disk(tokenized_dataset_path)

# Log a few examples from the tokenized dataset to ensure correctness
logging.info(f"Example from tokenized dataset: {tokenized_dataset['train'][0]}")

# Data collator to dynamically pad the sentences to the longest length in a batch during collation
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    "final-model-trainer",
    per_device_train_batch_size=1,  # Further reduced batch size
    per_device_eval_batch_size=1,   # Further reduced batch size
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=10,  # Log every 10 steps
)

# Evaluation function included in the trainer
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

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Simplified training loop for debugging
logging.info("Starting simplified training loop for debugging...")
try:
    # Get a single batch of training data
    for step, batch in enumerate(trainer.get_train_dataloader()):
        if step == 0:
            logging.info(f"Batch keys: {batch.keys()}")
            logging.info(f"Input IDs: {batch['input_ids'].shape}")
            logging.info(f"Attention Mask: {batch['attention_mask'].shape}")
            logging.info(f"Labels: {batch['labels'].shape}")
            outputs = model(**batch)
            logging.info(f"Model outputs: {outputs}")
        break
except Exception as e:
    logging.error(f"Error during training: {e}")

logging.info("Training process completed.")