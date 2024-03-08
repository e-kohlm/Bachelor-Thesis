import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification 
from datasets import load_dataset
from datasets import load_dataset_builder
import numpy as np
import codecs
from contextlib import redirect_stdout
import time


file_path = "../VUDENC_data/"
training_set = 'EXAMPLE_sql_dataset-TRAINING'

# Load data and get some information about it
train_dataset = load_dataset("json", data_files=file_path + training_set, split='train')
print(train_dataset)
print(train_dataset.features)
print(train_dataset[0])
print(train_dataset[-1])

# Working: from Hugging Face Tutorial
"""checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  # TODO: checkpoint = weights, which one?
tokenizer = AutoTokenizer.from_pretrained(checkpoint)   # TODO: it is not simply working with AutoAutoModelForQuestionAnswering
start_time = time.time()
i = 1
for snippet in train_dataset:
    print("\n############################################")
    print("i: ", i)  # num_rows: 2770 == i
    raw_inputs = train_dataset['code']
    '''with open('test_right_code.json', 'w') as f:    # überschreibt hier immer das ist okay
        with redirect_stdout(f):
            print("snippet['code']: ", snippet['code'], "\n*********\n")  
            print("snippet['label']: ", snippet['label'], "\n*********\n")'''
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") # pt means pytorch   
    #print("inputs: ", inputs, "\n############################################\n")
    i += 1
end_time = time.time()
elapsed_time = end_time - start_time
print("elapsed_time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))"""

# CodeT5
checkpoint = "Salesforce/codet5p-220m"
device = "cpu" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

# From Hugging Face working
"""inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# ==> print "Hello World"""

start_time = time.time()
i = 1
for snippet in train_dataset:
    print("\n############################################")
    print("i: ", i)  # num_rows: 2770 == i

    # not working: encodings = self._tokenizer.encode_batch(
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
    print("type: ", type(train_dataset['code']))

    inputs = tokenizer.encode(train_dataset['code'], return_tensors="pt").to(device)
    outputs = model.generate(inputs)   
    i += 1

end_time = time.time()
elapsed_time = end_time - start_time
print("elapsed_time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))






