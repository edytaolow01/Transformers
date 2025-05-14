from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

model_path = "save-from-32"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
input = ["Ale jestem zła", "This is a negative sentence"]

# Tokenize the input
tokenized_input = tokenizer(input, padding=True, truncation=True, return_tensors="pt")

# Perform inference
output = model(**tokenized_input)

# converts the raw prediction logits into final predicted class labels
preds = output.logits.argmax(dim=1)
print(preds)