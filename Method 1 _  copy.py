import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

raw_datasets = pd.read_csv("~/dev/transformer_practice/archive/Womens Clothing E-Commerce Reviews.csv")
#delete index column
raw_datasets = raw_datasets.iloc[:,1:]
#
for k in ["Title", "Review Text", "Division Name", "Department Name"]:
    raw_datasets[k].fillna("not specified", inplace= True)
for k in ["Clothing ID", "Age", "Rating", "Recommended IND", 'Positive Feedback Count']:
    raw_datasets[k].fillna(0, inplace= True)

#make sure all data is transformed to string
raw_datasets = raw_datasets.iloc[:, raw_datasets.columns != 'Class Name']
raw_datasets = raw_datasets.astype(str)
raw_train_datasets = raw_datasets.iloc[:10000]
raw_val_datasets = raw_datasets.iloc[10000:12000]
raw_val_datasets.reset_index( inplace=True, drop=True)
raw_test_datasets = raw_datasets.iloc[12000:14000]
raw_test_datasets.reset_index( inplace=True, drop=True)


#tokenization and get embedding
from sentence_transformers import SentenceTransformer
model_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

concatenated_table_train=[]#shape=( len(raw_train_datasets), len(raw_train_datasets.columns)*384 ))
concatenated_table_val=[]
concatenated_table_test=[]

for column in raw_train_datasets.columns:
    if column == "Clothing ID":
        concatenated_table_train = model_transformer.encode( raw_train_datasets[column] )
        concatenated_table_val = model_transformer.encode( raw_val_datasets[column] )
        concatenated_table_test = model_transformer.encode( raw_test_datasets[column] )
        continue
    clothing_train = model_transformer.encode( raw_train_datasets[column] )
    concatenated_table_train = np.concatenate((concatenated_table_train, clothing_train), axis=1)
    clothing_val = model_transformer.encode( raw_val_datasets[column] )
    concatenated_table_val = np.concatenate((concatenated_table_val, clothing_val), axis=1)
    clothing_test = model_transformer.encode( raw_test_datasets[column] )
    concatenated_table_test = np.concatenate((concatenated_table_test, clothing_test), axis=1)
    #print the shape of tokenized embedding
    print(concatenated_table_train.shape,concatenated_table_val.shape,concatenated_table_test.shape)

#save the tokenized dataframe
concatenated_table_train = pd.DataFrame(data = concatenated_table_train)
concatenated_table_val = pd.DataFrame(data = concatenated_table_val)
concatenated_table_test = pd.DataFrame(data = concatenated_table_test)
concatenated_table_train.to_csv('~/dev/transformer_practice/archive/train_string.csv', index= None)
concatenated_table_val.to_csv('~/dev/transformer_practice/archive/val_string.csv', index= None)
concatenated_table_test.to_csv('~/dev/transformer_practice/archive/test_string.csv', index= None)

#################################################################################################
#  To save time for debugging, I have two files in linux machine and I copy them.
#################################################################################################
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#size of the data frame: (4000, 3456) (1000, 3456) (1000, 3456)
raw_train_datasets = pd.read_csv("~/dev/transformer_practice/archive/train_string.csv")
raw_val_datasets = pd.read_csv("~/dev/transformer_practice/archive/val_string.csv")
raw_test_datasets = pd.read_csv("~/dev/transformer_practice/archive/test_string.csv")

#process the label
raw_datasets = pd.read_csv("~/dev/transformer_practice/archive/Womens Clothing E-Commerce Reviews.csv")
class_name = raw_datasets["Class Name"]
y = class_name.loc[:10000-1].astype(str)
from sklearn.preprocessing import LabelBinarizer
y_dense = LabelBinarizer().fit_transform(y)
print(y_dense)
from scipy import sparse
y_sparse = sparse.csr_matrix(y_dense)
print(y_sparse)

#training
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
x = raw_train_datasets
model = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x, y)

#predict
x2 = raw_val_datasets
preds = model.predict(x2)
print(preds.shape)

y2 = class_name.loc[10000:12000-1].astype(str)
print(y2.shape)

from sklearn.metrics import classification_report
print(classification_report(y2, preds))