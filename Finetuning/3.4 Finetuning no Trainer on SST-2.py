# Trainer makes finetuning, so now make finetuning manually
# 1) data preprocessing

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

raw_dataset = load_dataset("glue", "sst2")
checkpoint = "bert-base-uncased"
# pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# function for tokenizing all examples; takes two sentances and tokenize them
# truncation shorten tokenized sentances to required maximum
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

# tokenize the whole dataset using created function
# map. uses tokenize_function on every example 
# batch = True - it passes many examples to the function instead just one at a time
tokenized_datasets = raw_dataset.map(tokenize_function, batched=True) 

# adjusts padding to the longest sentance in batch
data_colletor = DataCollatorWithPadding(tokenizer=tokenizer)

# 2) postprocessing of tokenized_datasets
# model only wants numerical representation 
# dataloader - takes and input data
# Remove the columns corresponding to values the model does not expect (like the sentence1 and sentence2 columns) - model wants only numbers
# We only leave: input_ids - sequance of tokes corresponding to the words in the sentance, attention_mask: - which tokens are meaningful, which are just paddings, token_type_ids - which token belongs to which sentance (first -0, second - 1)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])

# Rename the column label to labels (because the model expects the argument to be named labels).
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# set format to the pytorch tensors instead of lists
tokenized_datasets.set_format("torch")

# check columns names
# print(tokenized_datasets.column_names)

# 3) define DataLoader
# DataLoader helps with easy and effective iteration per data in datasets
# Loads data in batches, shuffles data (random order of data, random shuffled and divided to batches) and uses function for processing batches
# DataLoader contains data divided into batches
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_colletor #needed to change collate_fn=data_collator to collate_fn=data_colletor
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_colletor
)

for batch in train_dataloader:
    break #break here prints only first round of loop
print({k: v.shape for k, v in batch.items()})

# the actual shapes will probably be slightly different since we set shuffle=True for the training dataloader and we are padding to the maximum length inside the batch

# 4) instantiate model

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# pass batch to this model to make sure that everything will go smoothly during training
outputs = model(**batch) #unpack batch dictionary as arguments passed separetly, tests if works properly (for one batch)
print(outputs.loss, outputs.logits.shape) #printed after one iteration because of break in for tensor 8x2 (8 examples, 2 logits (for 0 and 1))

# 5) optimizer and learning rate scheduler (ex. a linear decay from the maximum value (5e-5) to 0). To properly define it, we need to know the number of training steps we will take, which is the number of epochs we want to run multiplied by the number of training batches (which is the length of our training dataloader)

# Batch - number of examples taken once at a time 
# Epoch - iteration per all training examples; during 1 epoch, number of iteration depends on the batch and dataset size 1000/8 = 125 iteration

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader) #train_dataloader loades 8 examples per once (batch), 459 times = 3672 training examples
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0, #do not increase lr at the beginning
    num_training_steps=num_training_steps,
)
print(num_training_steps)

# work on GPU
model.to(device)

# 6) Training

progress_bar = tqdm(range(num_training_steps)) #to follow the progress

model.train() #set model for training settings

for epoch in range(num_epochs):
    for batch in train_dataloader: #process all following batches
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch) #uses current batch
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# 7) Evaluation

metric = evaluate.load("glue", "sst2")
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(): #do not compute gradient during evaluation
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"]) #accumulate batches

print(metric.compute()) #Once we have accumulated all the batches, we can get the final result   

