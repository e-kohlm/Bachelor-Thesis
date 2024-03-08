import torch
from transformers import pipeline
from datasets import load_dataset
from datasets import load_dataset_builder
import numpy as np
import codecs
from contextlib import redirect_stdout
from transformers import BertConfig, BertModel



# datasets from huggingface, docs: https://huggingface.co/docs/datasets/en/installation

# https://huggingface.co/learn/nlp-course/chapter3/2
# ok, output as expected
#raw_datasets = load_dataset("glue", "mrpc")
#print(raw_datasets)

"""file_path = "../VUDENC_data/"
plain_sql = "plain_sql"
finaltest_x = "sql_dataset_finaltest_X"
training = "EXAMPLE_sql_dataset-TRAINING"

# Ich habe alle files furchprobiert, überall dasselbe ich bekomme bytes, aber keine str (auch wenn ich r anstelle von rb als modus gebe
# TODO: das ist aber vielleicht auch gut so ... ? nur wie kann ich bytefiles laden in model, das ist die Fragen
finaltest_y = "sql_dataset_finaltest_Y"
finaltest_keys = "sql_dataset_keysfinaltest"
keystest = "sql_dataset_keystest"
keystrain = "sql_dataset_keystrain"

# TODO:
# okay, so weit so gut
#

# plain_sql is working, but not without 'train'
dataset = load_dataset("json", data_files=file_path + training)
train_dataset = dataset['train']
with open('test_huggingface_1.json', 'w') as f:
    with redirect_stdout(f):
        print(train_dataset[0])
        print(train_dataset[1])
        print(dataset['train'][-1])
        print("#####")
        print(dataset['train'])


dataset_one = load_dataset("json", data_files=file_path + training, split='train')
with open('test_huggingface:2.json', 'w') as f:
    with redirect_stdout(f):
        print(dataset_one)
        print("**********")
        print(dataset_one.features)"""



# TODO: irgendwo bei huggingface , evtl. bei finetunig, um daten kennenzulernen

#ds_builder = load_dataset_builder(file_path + finaltest_x, data_files=data_files)
#print("ds_builder: ", ds_builder)

#print(ds_builder.info.description)

#print(ds_builder.info.features)


"""# Building the config
config = BertConfig()

# Building the model from the config
#model = BertModel(config)

model = BertModel.from_pretrained("bert-base-cased")
print(config)"""


# pipelines
"""Not working: 
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended."""

classifier = pipeline("sentiment-analysis")
output = classifier("I've been waiting for a HuggingFace course my whole life.")
print(output)

question_answerer = pipeline("question-answering")
output = question_answerer(
    question="I this code vulnerable?",
    context="with open('test_huggingface_1.json', 'w') as f: with redirect_stdout(f):"
            " print(train_dataset[0]) print(train_dataset[1]) print(dataset['train'][-1]) '])"
)
print(output)


classifier = pipeline("zero-shot-classification")
output = classifier(
    "with open('test_huggingface_1.json', 'w') as f: with redirect_stdout(f):"
    " print(train_dataset[0]) print(train_dataset[1]) print(dataset['train'][-1]) '])",
    candidate_labels=["vulnerable code", "not vulnerable code"],
)
print(output)



# sql_dataset_finaltest_X ist vom Type: Binary(application/octet-stream)
# https://www.geeksforgeeks.org/reading-binary-files-in-python/
# it is working with all files, but only in rb mode, I cannot convert into string
# Open the binary file
"""file = open(file_path + keystest, "rb")
# Reading the first three bytes from the binary file
data = file.read(3)
# Knowing the Type of our data
print("type: ", type(data))
# Printing data by iterating with while loop
while data:
    #print("1: ", data)
    data = file.read(3)
    #print("2: ", data)
    #print("3: ", str(data))
    print("4: ", list(data))  # Gibt mir Liste mit je 3 integer
# Close the binary file
file.close()"""

# Program for converting bytes to string using decode()
# TODO: not working
#file = open(file_path + finaltest_x, "rb")
#data = file.read(3)
# converting
#output = data.decode()  # UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
#output = str(data, 'UTF-8')  # UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
#output = codecs.decode(data)
#output = str(data, 'UTF-8')
# display output
#print('\nOutput:', output)
#print(type(output))

# List of encodings to try not working
"""encodings_to_try = ['utf-8', 'utf-16', 'ascii']  # Work with rectangle: ['ISO 8859-1', 'latin-1', ]
while data:
    for encoding in encodings_to_try:
            try:
                output = data.decode(encoding)
                print("Decoded using {} encoding: {}".format(encoding, output))
                print('Output:', output)
                print(type(output))
                break
            except UnicodeDecodeError:
                print("Failed to decode using {} encoding".format(encoding))

file.close()"""




# TODO: weiß nicht mehr ob es geklappt hat
# Open the file in binary mode
"""with open(file_path + finaltest_x, 'rb') as f:
    # Read the data into a NumPy array
    array = np.fromfile(f, dtype=np.uint8)  # Change dtype according to your data
    print("\n4: ", array[0], "\n")
file.close()"""



