# Text summarization on polish news dataset

from datasets import load_dataset
import pprint
import pandas as pd
from transformers import AutoTokenizer
from transformers import pipeline
from random import randrange        
import torch

# choose GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

raw_dataset = load_dataset("WiktorS/polish-news")

# check if dataset has any splits
# print(raw_dataset)
# train: Dataset({features: ['link', 'title', 'headline', 'content'], num_rows: 248123

num_sample = 43

# check single sample
# pprint.pprint(raw_dataset["train"][num_sample])

# check the features

#features = raw_dataset["train"].features
#pprint.pprint(features)

# create convenient panda frame
df_polish_news = pd.DataFrame(data=raw_dataset["train"], columns=raw_dataset["train"].features.keys())

pprint.pprint(df_polish_news.describe())

# Longest content
# longest_content_summary_id = df_polish_news["content"].str.len().argmax()
# longest_content_summary_num_char = df_polish_news["content"].str.len().max()
# longest_content_summary = df_polish_news["content"][longest_content_summary_id]
#print(f"The id of the longest content is: {longest_content_summary_id}")
#print(f"It has a total of {longest_content_summary_num_char}")
#print(f"Longest content: {longest_content_summary}")

# mean_summary_review_chars = df_polish_news["content"].str.len().mean()
# print(f"On average the review_summary field has {mean_summary_review_chars} characters")


# delete rows with content None type
df_polish_news = df_polish_news[df_polish_news["content"].notna()]

# Check the format of each cell in the 'content' column
invalid_entries = []

for index, content in df_polish_news["content"].items():
    if not isinstance(content, str):
        invalid_entries.append((index, type(content), content))

# Print out invalid entries if any
if invalid_entries:
    print(f"Found {len(invalid_entries)} invalid entries in the 'content' column:")
    for entry in invalid_entries:
        print(f"Index: {entry[0]}, Type: {entry[1]}, Content: {entry[2]}")
else:
    print("All entries in the 'content' column are valid strings.")


#first by pipeline 

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


sample_size = 10

# inference

for i in range(sample_size):
    try:
        result = summarizer(df_polish_news.iloc[i]['content'][:1500], max_length=130, min_length=30, do_sample=False) #needed to shorten the lenght of the content to make model work
        print(f"Original content {i+1}: {df_polish_news.iloc[i]['content']}")
        print(f"Summary {i+1}: {result[0]['summary_text']}\n")
    except Exception as e:
        print(f"Error summarizing content {i+1}: {e}\n")

# evaluation
