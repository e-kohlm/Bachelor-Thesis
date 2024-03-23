import VUDENC_utils
import sys
import os.path
import json
from datetime import datetime
import random
import pickle
import numpy
from contextlib import redirect_stdout
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.preprocessing import sequence
# from keras import backend as K
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.utils import class_weight
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors

# TODO: von VUDENC geklaut, allerdings wird hier nicht makemodel gemacht. Insofern solle ich den vielleicht umbenennen
# TODO: A bit of
"""
This code was first implemented in VUDENC, a few changes are made:
- the code snippets are stored differently, so they can be used with a Hugging Face model for finetuning.
  The data itself has not been changed.  
- But there is no w2v vectorization made, since the transformer model itself takes care of it.
- some comments were added or rather a few changes to variables etc. if it helps for understanding
- unused code was removed
"""




###main####
# default mode / type of vulnerability
mode = "sql"

# get the vulnerability from the command line argument
if (len(sys.argv) > 1):
    mode = sys.argv[1]

progress = 0
count = 0

### paramters for the filtering and creation of samples
restriction = [20000, 5, 6, 10]  # which samples to filter out
step = 5  # step lenght n in the description
fulllength = 200  # context length m in the description

# TODO: Start with the original plain_sql, download it again before submitting!

# load data
with (open('../VUDENC_data/plain_' + mode, 'r') as infile):
    data = json.load(infile)

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("finished loading. ", nowformat)

# Vulnerable code snippets are distinguished from not vulnerable code snippets and labeled accordingly

allblocks = []
snippet_id = 0      # TODO: das ist auch das, was Laura als snippet bezeichnet, oder?
i = 0               # TODO: Just for the example, the feasibility test, remove later
for repository in data:
    i += 1          # TODO: Just for the example, the feasibility test, remove later
    if i < 4:       # TODO: Just for the example, the feasibility test, remove later
    
        progress = progress + 1
        print("\nprogress: ", progress)
        print("repo: ", repository)

        for commit in data[repository]:
            print("commit: ", commit)

            if "files" in data[repository][commit]:
                for f in data[repository][commit]["files"]:
                    if not "source" in data[repository][commit]["files"][f]:
                        continue
                    if "source" in data[repository][commit]["files"][f]:
                        sourcecode = data[repository][commit]["files"][f]["source"]

                        allbadparts = []
                        for change in data[repository][commit]["files"][f]["changes"]:
                            # get the modified or removed parts from each change that happened in the commit
                            badparts = change["badparts"]
                            count = count + len(badparts)

                            for bad in badparts:
                                pos = VUDENC_utils.findposition(bad, sourcecode)
                                if not -1 in pos:
                                    allbadparts.append(bad)
                        if len(allbadparts) > 0:
                            positions = VUDENC_utils.findpositions(allbadparts, sourcecode)

                            # get the file split up in samples
                            blocks = VUDENC_utils.getblocks(sourcecode, positions, step, fulllength)  # TODO: das muss ich auch mit Code machen, den ich testen will
                            #print("blocks: ", blocks)
                            for b in blocks:  # each block is a tuple of code and label
                                """
                                In VUDENC a block was stored in a list, now it is stored in a dictionary 
                                """
                                block_dict = {}
                                """with open('../test_outputs/test_one_block.json', 'w') as f:
                                    with redirect_stdout(f):
                                        print("b: ", b)"""
                                # Save each code snippet with its label (vulnerable = 1, not vulnerable = 0) in a dict
                                #block_dict['snippet_id'] = snippet_id  # TODO: weg damit, es wird später ohnehin neu geshuffelt. I guess
                                block_dict['code'] = b[0]
                                block_dict['labels'] = b[1]
                                allblocks.append(block_dict)
                                snippet_id += 1


with open('test_allblocks.json', 'w') as f:
    with redirect_stdout(f):
        print(allblocks)


# ab hier: Data in Form bringen

############## meins######
##### allblocks [] ist der komplett code nach filtering

#######################################WORKING BUT NOT NECESSARY#########################

keys = []

# randomize the sample and split into train, validate and final test set
print("snippet_id: ", snippet_id)
print("len allblocks: ", len(allblocks)) #len allblocks:  284599
for i in range(len(allblocks)):
    keys.append(i)

random.shuffle(keys)

cutoff = round(0.7 * len(keys))  # 70% for the training set
cutoff2 = round(0.85 * len(keys))  # 15% for the validation set and 15% for the final test set

keystrain = keys[:cutoff]  # Keys der Trainingsdaten
keysvalidation = keys[cutoff:cutoff2]  # Keys der Testdaten
keysfinaltest = keys[cutoff2:]  # Keys des final test

print("cutoff " + str(cutoff))  # TODO: VUDENC auf gruenau: 199219 - und auch at home
print("cutoff2 " + str(cutoff2))  # TODO: VUDENC auf gruenau: 241909 - und auch at home


# Save keys of three datasets to file
with open('../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset_keystrain', 'w') as fp:
    fp.write(str(keystrain))
with open('../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset_keysvalidation', 'w') as fp:
    fp.write(str(keysvalidation))
with open('../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset_keysfinaltest', 'w') as fp:
    fp.write(str(keysfinaltest))

training_set = []
validation_set = []
test_set = []

print("Creating training dataset... (" + mode + ")")
for k in keystrain:
    block = allblocks[k]
    training_set.append(block)  # TODO: hier wird immer nur snippet id 3956 angehängt

print("Creating validation dataset...")
for k in keysvalidation:
    block = allblocks[k]
    validation_set.append(block)

print("Creating finaltest dataset...")
for k in keysfinaltest:
    block = allblocks[k]
    test_set.append(block)

print("Train length: " + str(len(training_set)))  #
print("Validation length: " + str(len(validation_set))) #593
print("Testing length: " + str(len(test_set)))  # 594

now = datetime.now()
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)
######################################################

# saving samples
with open('../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset-TRAINING', 'w') as fp:
    fp.write(json.dumps(training_set))

with open('../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset-VALIDATION', 'w') as fp:
    fp.write(json.dumps(validation_set))
    
with open('../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset-TESTING', 'w') as fp:
    fp.write(json.dumps(test_set))

"""# Nur für jetzt, damit ich selbe Länge habe
with open('../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset', 'w') as fp:
    fp.write(json.dumps(training_set))   """ 

######## BIS hier für splitting daten


"""# ohne splitting geht  das dann wohl einfach so: 
with open('../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset', 'w') as fp:
    fp.write(json.dumps(allblocks))"""
