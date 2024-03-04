import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from datasets import load_dataset_builder
import numpy as np
import codecs
from contextlib import redirect_stdout


file_path = "../VUDENC_data/"
training_set = 'EXAMPLE_sql_dataset-TRAINING'

# Load data and get some information about it
train_dataset = load_dataset("json", data_files=file_path + training_set, split='train')
print(train_dataset)
print(train_dataset.features)
print(train_dataset[0])
print(train_dataset[-1])


