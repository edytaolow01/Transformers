from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import torch
import evaluate

# choose GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load benchmark for sentiment analysis (Binary classification)
raw_datasets = load_dataset("glue", "sst2")

# choose model
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# function for tokanizing every sentence
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

# tokanize all dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# dynamic padding (to the lenght of the maximum sentance in the batch)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# define the training arguments for the Trainer (include the directory to save the trained model and other hyperparameters)
training_args = TrainingArguments("sst2-finetuned-model")

# define the model to be fine-tuned
# AutoModelForSequenceClassification is a class from the transformers library designed for sequence classification tasks. When you use this class with from_pretrained, it automatically configures the model architecture to fit the sequence classification task.
# num_labels=2 specifies that the model will be fine-tuned for a binary classification task (i.e., two classes).
# load bert architecture and weights for classification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "sst2")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# define the Trainer using the model, training arguments, datasets, data collator, and tokenizer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Define custom metrics function to evaluate the model’s performance during training
    tokenizer=tokenizer,
)

trainer.train()

# uses the Trainer instance to make predictions on the validation dataset
predictions = trainer.predict(tokenized_datasets["validation"])

# prints the shape of the raw predictions array. The shape is typically (number_of_samples, number_of_labels), where each element represents the model's predicted logits for each class
print(predictions.predictions.shape, predictions.label_ids.shape)

# converts the raw prediction logits into final predicted class labels
preds = np.argmax(predictions.predictions, axis=-1)

print(preds)

model.save_pretrained("save-from-32")