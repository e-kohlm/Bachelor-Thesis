# Code from Video
"""from transformers import pipeline

#classifier = pipeline("sentiment-analysis")
#res = classifier("I've been waiting for a HuggingFace course for my whole life.")
#print(res)

generator = pipeline("text-generation", model="distilgpt2")

response = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,)
print(response)"""
import pickle
import numpy
from datetime import datetime
import json
from contextlib import redirect_stdout
from itertools import islice

# Code from CodeT5+ github README

"""from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

checkpoint = "Salesforce/instructcodet5p-16b"
device = "cpu" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True,
                                              trust_remote_code=True).to(device)

encoding = tokenizer("def print_hello_world():", return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=15)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))"""

# Code from Huggingface https://huggingface.co/Salesforce/codet5p-220m
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

checkpoint = "Salesforce/codet5p-220m"
device = "cpu"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

# inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
# outputs = model.generate(inputs, max_length=10)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# from ChatGPT
# Define the input prompt indicating the task to the model
prompt = "Find and point out any vulnerable code snippets in the following Python code:\n"

# Define the input Python code containing potential vulnerabilities
file_path = "../VUDENC_data/"
mode = 'sql'


# plain_sql: working so far that it opens file and writes some data into another file
with open(file_path + 'plain_sql', 'r') as input_file:
    print("type of plain_sql: ", type(input_file))  
    data = json.load(input_file)

x = 1
y = 2
for r in islice(data, x):   
    for c in islice(data, y):
        with open('my_test_data_1.txt', 'w') as f:
            with redirect_stdout(f): 
                print("r: ", r, "\n")
                print("c: ", c, "\n")
                print("data: ", data, "\n")
repo_count = 0
for r in data:
    repo_count += 1
    print("\nr: ", r)   
    for c in data[r]:        
        print("c: ", c)              

print("repo_count plain_sql: ", repo_count)        

now = datetime.now()  # current date and time
start_time = now.strftime("%H:%M")
print("Start:", start_time)

# sql_dataset_finaltest_x not working
with open(file_path + 'sql_dataset_finaltest_X', 'rb') as in_file:  # FIXME: Error with r or rb
    print("type of sql_dataset_finaltest_X: ", type(in_file))  
    # FIXME: UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
    data = json.load(in_file)  
    

x = 1
y = 2
for r in islice(data, x):   
    for c in islice(data, y):
        with open('my_test_data_2.txt', 'w') as f:
            with redirect_stdout(f): 
                print("r: ", r, "\n")
                print("c: ", c, "\n")
                print("data: ", data, "\n")
repo_count = 0
for r in data:
    repo_count += 1
    print("\nr: ", r)   
    for c in data[r]:        
        print("c: ", c)              

print("repo_count: ", repo_count)    
