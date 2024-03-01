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
from datetime import datetime

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

with open(file_path + mode + '_dataset_finaltest_X', 'rb') as file:
    FinaltestX = pickle.load(file)
with open(file_path + mode + '_dataset_finaltest_Y', 'rb') as file:
    FinaltestY = pickle.load(file)

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("nowformat:", nowformat)

# Encode the input prompt and code using the tokenizer
input_sequence = prompt + FinaltestX + FinaltestY
input_sequence_tokens = tokenizer.encode(input_sequence, return_tensors="pt").to(device)

# Generate the output code prediction using the model
output_sequence = model.generate(input_sequence_tokens, max_length=150)

# Decode and print the output sequence
decoded_output = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
print(decoded_output)
