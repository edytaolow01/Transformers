# Text summarization on polish news dataset

from datasets import load_dataset
from transformers import AutoTokenizer    
import torch
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# choose GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load dataset
# dataset contains only a "train" split, so we need to split this
# train: Dataset({features: ['link', 'title', 'headline', 'content'], num_rows: 248123
raw_dataset = load_dataset("WiktorS/polish-news", split="train").shuffle(seed=42).select(range(100)) # choses which part of split to load #select 100 random
raw_dataset = raw_dataset.train_test_split(test_size=0.1) # split="train" - necessary to load in the purpose of future splitting

# dataset preprocessing: delete None Types in headline and content columns

def filter_none(example):
    return example['headline'] is not None and example['content'] is not None

raw_dataset = raw_dataset.filter(filter_none)

# choose model

checkpoint="facebook/bart-large-cnn"

# tokenize

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# tokenization function (mapping)

# Prefix the input with a prompt so model knows this is a summarization task. Some models capable of multiple NLP tasks require prompting for specific tasks.
prefix = "summarize: "

def tokenize_function(example):
    
    inputs = [prefix + doc for doc in example["content"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True) #dictionary with keys: input_ids, attention_mask
    #tokenize not only the input content but also labels
    labels = tokenizer(text_target=example["headline"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"] # add labels to the dictionary
    return model_inputs # return dictionary with keys: input_ids, attention_mask, labels


tokenize_dataset = raw_dataset.map(tokenize_function, batched=True) #features: ['link', 'title', 'headline', 'content', 'input_ids', 'attention_mask', 'labels']
print(tokenize_dataset["train"][0])

# create a batch of examples; dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length
# data collator to load batch into tokanization function

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# arguments for trainer

training_args = Seq2SeqTrainingArguments(
    output_dir="final-model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
)

# evaluation fuction included in the trainer. Including a metric during training is often helpful for evaluating your model’s performance. 

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True) #Converts tokens to human-readable text, skipping special tokens.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) #Replaces -100 values in labels with pad tokens so they can be decode
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) 

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True) #Computes the ROUGE metric for the model's predictions

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions] #Counts the number of non-pad tokens in each prediction
    result["gen_len"] = np.mean(prediction_lens) #Calculates the average number of non-pad tokens in the predictions

    return {k: round(v, 4) for k, v in result.items()} #Returns the ROUGE metric results rounded to four decimal places

# initialize the model

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


# trainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenize_dataset["train"],
    eval_dataset=tokenize_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)

# train model

trainer.train()

# test model on new data


