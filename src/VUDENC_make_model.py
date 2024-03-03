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

# TODO: von VUDENC geklaut, insofern sollt ich das villeicht auch in eine extra dir: VUDENC_src packen

"""
run from /src with command nice -n 19 python3 VUDENC_make_model.py sql
Running without error with sql as example
Input files: 
plain_sql
and if w2v is needed:
word2vec_withString10-100-200.model
word2vec_withString10-100-200.model.syn1neg.npy
word2vec_withString10-100-200.model.wv.vectors.npy

Created are the files:  
- sql_dataset_finaltest_X
- sql_dataset_finaltest_Y
- sql_dataset_keysfinaltest
- sql_dataset_keystest
- sql_dataset_keystrain
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


# load data
with (open('../VUDENC_data/plain_' + mode, 'r') as infile):  # Input if mode=sql, dann der File plain_sql from Code/data usw.
    data = json.load(infile)
    # print("data: ", data) # Daten sind da

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("finished loading. ", nowformat)

### meins ###### filtering of unwanted code, I think
allblocks = []

"""print("type data: ", type(data))             # it is a dict
with open('keys_plain_sql.txt', 'w') as f:      # keys sind die repositories, also r
    with redirect_stdout(f):
        print("keys: ", data.keys())"""


b_dict = {}
i = 0               # TODO: weg
for r in data:  # repository
    i += 1          # TODO: rausnehmen,nur zum testen, damit ich kleinere Datensätze bekommen!!!
    if i < 4:      # TODO: weg
    
        progress = progress + 1
        print("\nprogress: ", progress)
        print("r: ", r)

        for c in data[r]:  # commit
            print("c: ", c)

            if "files" in data[r][c]:            
                #  if len(data[r][c]["files"]) > restriction[3]:
                # too many files
                #    continue
                # print("files: ", data[r][c]) # hier kommt was
                for f in data[r][c]["files"]:   # files f = xxx.py
                    #print("f: ", f)

                    #      if len(data[r][c]["files"][f]["changes"]) >= restriction[2]:
                    # too many changes in a single file
                    #       continue

                    if not "source" in data[r][c]["files"][f]:
                        #print("no sourcecode")
                        # no sourcecode
                        continue

                    if "source" in data[r][c]["files"][f]:
                        sourcecode = data[r][c]["files"][f]["source"]
                        #print("sourcecode", sourcecode)
                        #     if len(sourcecode) > restriction[0]:
                        # sourcecode is too long
                        #       continue

                        allbadparts = []

                        for change in data[r][c]["files"][f]["changes"]:

                            # get the modified or removed parts from each change that happened in the commit
                            badparts = change["badparts"]
                            # print("count y: ", count)
                            count = count + len(badparts)
                            # print("count x: ", count)

                            #     if len(badparts) > restriction[1]:
                            # too many modifications in one change
                            #       break

                            for bad in badparts:
                                # print("bad ", bad)
                                # check if they can be found within the file
                                pos = VUDENC_utils.findposition(bad, sourcecode)
                                #print("pos ", pos)
                                if not -1 in pos:
                                    # print("pos ", pos)
                                    allbadparts.append(bad)
                        if len(allbadparts) > 0:
                            #print("XXXX:", len(allbadparts))
                            #   if len(allbadparts) < restriction[2]:
                            # find the positions of all modified parts
                            positions = VUDENC_utils.findpositions(allbadparts, sourcecode)
                            #print("positions: ", positions)
                            # get the file split up in samples
                            blocks = VUDENC_utils.getblocks(sourcecode, positions, step, fulllength)
                            """with open('blocks.json', 'w') as f:
                                with redirect_stdout(f):
                                    print("r: ", r)
                                    print("c: ", c)
                                    print("f: ", f)
                                    print("blocks: ", blocks)"""

                            for b in blocks:
                                with open('one block.json', 'w') as f:
                                    with redirect_stdout(f):
                                        print("b: ", b)     # das ist eine Liste: b[0] ist Code, b[1] ist Label
                                                            # each is a tuple of code and label
                                # meins: turn into dictionary
                                b_dict['code'] = b[0]
                                b_dict['label'] = b[1]
                                allblocks.append(b_dict)


# bis hier: Data labeling in vulnerable, not vulnerable


with open('allblocks.json', 'w') as f:
    with redirect_stdout(f):
        print(allblocks)


# ab hier: Data in Form bringen

############## meins######
##### allblocks [] ist der komplett code nach filtering
# print("type allblocks: ", type(allblocks)) # allblocks ist eine Liste fon Listen

keys = []

# randomize the sample and split into train, validate and final test set
print("len allblocks: ", len(allblocks)) #len allblocks:  284599
for i in range(len(allblocks)):
    #print("i, allblocks[i]: ", i, ",", allblocks[i])
    keys.append(i)
random.shuffle(keys)

cutoff = round(0.7 * len(keys))  # 70% for the training set
cutoff2 = round(0.85 * len(keys))  # 15% for the validation set and 15% for the final test set

keystrain = keys[:cutoff]  # Keys der Trainingsdaten
keysvalidation = keys[cutoff:cutoff2]  # Keys der Testdaten
keysfinaltest = keys[cutoff2:]  # Keys des final test

print("cutoff " + str(cutoff))  # TODO: VUDENC auf gruenau: 199219 - und auch at home
print("cutoff2 " + str(cutoff2))  # TODO: VUDENC auf gruenau: 241909 - und auch at home


# jeweils eine Liste von keys, sonst nichts
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset_keystrain', 'w') as fp:
    # The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream,
    #pickle.dump(keystrain, fp)  # pickle module not considered secure anymore
    fp.write(str(keystrain))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset_keysvalidation', 'w') as fp:
    #pickle.dump(keystest, fp)
    fp.write(str(keysvalidation))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset_keysfinaltest', 'w') as fp:
    #pickle.dump(keysfinaltest, fp)
    fp.write(str(keysfinaltest))

training_set = []
validation_set = []
test_set = []

print("Creating training dataset... (" + mode + ")")
for k in keystrain:
    block = allblocks[k]
    training_set.append(block)

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
now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)
######################################################

# saving samples
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset-TRAINING', 'w') as fp:
    fp.write(json.dumps(training_set))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset-VALIDATION', 'w') as fp:
    fp.write(json.dumps(validation_set))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset-TESTING', 'w') as fp:
    fp.write(json.dumps(test_set))
