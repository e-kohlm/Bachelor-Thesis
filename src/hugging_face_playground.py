import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from datasets import load_dataset_builder
import numpy as np
import codecs


# datasets from huggingface, docs: https://huggingface.co/docs/datasets/en/installation

# https://huggingface.co/learn/nlp-course/chapter3/2
# ok, output as expected
#raw_datasets = load_dataset("glue", "mrpc")
#print(raw_datasets)

file_path = "../VUDENC_data/"
plain_sql = "plain_sql"
finaltest_x = "sql_dataset_finaltest_X"

# Ich habe alle files furchprobiert, überall dasselbe ich bekomme bytes, aber keine str (auch wenn ich r anstelle von rb als modus gebe
# TODO: das ist aber vielleicht auch gut so ... ? nur wie kann ich bytefiles laden in model, das ist die Fragen
finaltest_y = "sql_dataset_finaltest_Y"
finaltest_keys = "sql_dataset_keysfinaltest"
keystest = "sql_dataset_keystest"
keystrain = "sql_dataset_keystrain"

# TODO:
#  - make_model anschauen, was genau wurde da gemacht mit plain_sql?
#     Umwandlung in numpy array, dann kann da kein String rauskommen
#     mit w2v representation ersetzt? dann kann da kein string rauskommne
#  - was brauche ich? plain test mit einem label? wie bekomme ich das?

# plain_sql is working, but not without 'train',
#dataset = load_dataset("json", data_files=file_path + plain_sql)
#print(dataset['train'][0])
#print(dataset['train'][-1])
#dataset_one = load_dataset("json", data_files=file_path + plain_sql, split='train')
#print(dataset_one[0])

# sql_dataset_finaltest_X ist vom Type: Binary(application/octet-stream)
# https://www.geeksforgeeks.org/reading-binary-files-in-python/
# it is working with all files, but only in rb mode, I cannot convert into string
# Open the binary file
file = open(file_path + keystest, "rb")
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
file.close()

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



# TODO: irgendwo bei huggingface , evtl. bei finetunig, um daten kennenzulernen

#ds_builder = load_dataset_builder(file_path + finaltest_x, data_files=data_files)
#print("ds_builder: ", ds_builder)

#print(ds_builder.info.description)

#print(ds_builder.info.features)
