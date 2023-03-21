import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer

read_data = pd.read_csv('~/dev/transformer_practice/archive/Womens Clothing E-Commerce Reviews.csv')

# filling missing data
for k in ["Title", "Review Text", "Division Name", "Department Name", "Class Name"]:
    read_data[k].fillna("not specified", inplace=True)
for k in ["Clothing ID", "Age", "Rating", "Recommended IND", 'Positive Feedback Count']:
    read_data[k].fillna(0, inplace=True)
del read_data['Unnamed: 0']
read_data = read_data.astype(str)
# print(read_data[:3])

# combine all columns to one column, keep class name
for column in read_data.columns:
    if column == "Clothing ID":
        read_data['combined'] = read_data['Title'] + '. '
        continue
    if column == "Class Name":
        continue
    read_data['combined'] = read_data['combined'] + read_data[column] + '. '
read_data.drop(columns=["Title", "Review Text", "Division Name", "Department Name",
                        "Clothing ID", "Age", "Rating", "Recommended IND", 'Positive Feedback Count'], axis=1,
               inplace=True)
# print(read_data[0:3])

# keep the order of features
read_data_training = read_data[:12000]
read_data_validation = read_data[12000:1300]
read_data_validation.reset_index(inplace=True, drop=True)

# change categorical labels to interger
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
read_data_training['Class Name'] = le.fit(read_data['Class Name']).transform(read_data_training['Class Name'])
read_data_validation['Class Name'] = le.fit(read_data['Class Name']).transform(read_data_validation['Class Name'])
print(read_data_validation[:3])

# save the datasets and load them using load_dataset
read_data_training.to_csv('~/dev/transformer_practice/archive/Womens Clothing E-Commerce Reviews train.csv', index=None)
read_data_validation.to_csv('~/dev/transformer_practice/archive/Womens Clothing E-Commerce Reviews validation.csv',
                            index=None)
data_files = {'train': 'transformer_practice/archive/Womens Clothing E-Commerce Reviews train.csv',
              "validation": "transformer_practice/archive/Womens Clothing E-Commerce Reviews validation.csv", }
raw_datasets = load_dataset("csv", data_files=data_files)
print(raw_datasets.column_names)

# tokenization
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["combined"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(tokenized_datasets)

##training preparation
# delete original string columns "combined"
tokenized_datasets = tokenized_datasets.remove_columns(["combined"])
tokenized_datasets = tokenized_datasets.rename_column("Class Name", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

# dataloader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=len(read_data_validation),
                             collate_fn=data_collator)

# model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=len(np.unique(read_data["Class Name"])))
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# optimizer
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from accelerate import Accelerator

accelerator = Accelerator()
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

# dynamic learning rate
from transformers import get_scheduler

num_epochs = 30
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# progress bar
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

# training
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        progress_bar.update(1)

    # evaluation
    from sklearn.metrics import classification_report

    model.eval()

    for batch in eval_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        print(classification_report(batch['labels'], predictions))
