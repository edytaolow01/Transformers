from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import torch

# choose GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load dataset
raw_datasets = load_dataset("glue", "mrpc")

# choose model
checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# function for tokanizing every sentence
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# tokanize all dataset
# map function: 
# The results of the function are cached, so it won't take any time if we re-execute the code
# It can apply multiprocessing to go faster than applying the function on each element of the dataset
# It does not load the whole dataset into memory, saving the results as soon as one element is processed.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# data_collator puts together all the samples in a batch.
# dynamic padding (to the lenght of the maximum sentance in the batch)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

preds = np.argmax(predictions.predictions, axis=-1)

print("done")