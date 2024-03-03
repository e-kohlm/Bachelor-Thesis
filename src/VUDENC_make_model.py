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

#mode2 = str(step) + "_" + str(fulllength)

# TODO: w2v Zeug brauche ich nicht, oder doch?
# TODO: Wenn nein: 3 word2vec files wegschmeißen
# TODO: Wenn ja: in ein Unterverziechnis w2v packen und im Code ändern

### hyperparameters for the w2v model
mincount = 10  # minimum times a word has to appear in the corpus to be in the word2vec model
iterationen = 100  # training iterations for the word2vec model
s = 200  # dimensions of the word2vec model
w = "withString"  # word2vec model is not replacing strings but keeping them

# get word2vec model
w2v = "word2vec_" + w + str(mincount) + "-" + str(iterationen) + "-" + str(s)
w2vmodel = "../VUDENC_data/" + w2v + ".model"  # das ist das bereits erstellte model mit den o.g. parametern, also die Datei:
# word2vec_withString-10-100-200.model
# hier im Project habe ich das von mir trainierte w2v, falls ich neu trainieren muss
# zum testen reicht aber vielleicht auch das alte (vom 10.12.????

# load word2vec model
if not (os.path.isfile(w2vmodel)):
    print("word2vec model is still being created...")
    sys.exit()

#w2v_model = Word2Vec.load(w2vmodel)
#word_vectors = w2v_model.wv

# load data
with (open('../VUDENC_data/plain_' + mode,
           'r') as infile):  # Input if mode=sql, dann der File plain_sql from Code/data usw.
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
                                allblocks.append(b)




with open('allblocks.json', 'w') as f:
    with redirect_stdout(f):
        print("allblocks: ", allblocks)
        
############## meins######
##### allblocks [] ist der komplett code nach filtering
# print("type allblocks: ", type(allblocks)) # allblocks ist eine Liste fon Listen

keys = []

# randomize the sample and split into train, validate and final test set
print("len allblocks: ", len(allblocks)) #len allblocks:  284599
for i in range(len(allblocks)):
    keys.append(i)
random.shuffle(keys)

cutoff = round(0.7 * len(keys))  # 70% for the training set
cutoff2 = round(0.85 * len(keys))  # 15% for the validation set and 15% for the final test set

keystrain = keys[:cutoff]  # Keys der Trainingsdaten
keystest = keys[cutoff:cutoff2]  # Keys der Testdaten
keysfinaltest = keys[cutoff2:]  # Keys des final test

print("cutoff " + str(cutoff))  # TODO: VUDENC auf gruenau: 199219 - und auch at home
print("cutoff2 " + str(cutoff2))  # TODO: VUDENC auf gruenau: 241909 - und auch at home


# jeweils eine Liste von keys, sonst nichts
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset_keystrain', 'w') as fp:
    # The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream,
    #pickle.dump(keystrain, fp)  # pickle module not considered secure anymore
    fp.write(str(keystrain))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset_keystest', 'w') as fp:
    #pickle.dump(keystest, fp)
    fp.write(str(keystest))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset_keysfinaltest', 'w') as fp:
    #pickle.dump(keysfinaltest, fp)
    fp.write(str(keysfinaltest))

TrainX = []
TrainY = []
ValidateX = []
ValidateY = []
FinaltestX = []
FinaltestY = []

###### Für alle Daten, verteilt in Training, Test, finaltest, werden hier für alle keys, die w2v Werte geholt und in einer Liste gespeichert
print("Creating training dataset... (" + mode + ")")
for k in keystrain:
    block = allblocks[k]
    
    TrainX.append(block[0])  # append the list of vectors to the X (independent variable)
    TrainY.append(block[1])  # append the label to the Y (dependent variable)    
    #print("TrainY: ", TrainY)  # Elke war das

print("Creating validation dataset...")
for k in keystest:
    block = allblocks[k]
    #code = block[0]
    #token = VUDENC_utils.getTokens(code)  # get all single tokens from the snippet of code
    #vectorlist = []
    '''for t in token:  # convert all tokens into their word2vec vector representation  # TODO: Möchte ich das???
        if t in word_vectors.key_to_index and t != " ":
            vector = w2v_model.wv[t]
            vectorlist.append(vector.tolist())'''
    ValidateX.append(block[0])  # append the list of vectors to the X (independent variable)
    ValidateY.append(block[1])  # append the label to the Y (dependent variable)

print("Creating finaltest dataset...")
for k in keysfinaltest:
    block = allblocks[k]
    #code = block[0]
    #token = VUDENC_utils.getTokens(code)  # get all single tokens from the snippet of code
    #vectorlist = []
    '''for t in token:  # convert all tokens into their word2vec vector representation  # TODO: Möchte ich das???
        if t in word_vectors.key_to_index and t != " ":
            vector = w2v_model.wv[t]
            vectorlist.append(vector.tolist())'''
    FinaltestX.append(block[0])  # append the list of vectors to the X (independent variable)
    FinaltestY.append(block[1])  # append the label to the Y (dependent variable)

print("Train length: " + str(len(TrainX)))
print("Test length: " + str(len(ValidateX)))
print("Finaltesting length: " + str(len(FinaltestX)))
now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)
######################################################

# saving samples

with open('../VUDENC_data/' + 'elke_' + mode + '_dataset-train-X_', 'w') as fp:
    #  pickle.dump(TrainX, fp)
    fp.write(str(TrainX))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset-train-Y_', 'w') as fp:
    #  pickle.dump(TrainY, fp)
    fp.write(str(TrainY))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset-validate-X_', 'w') as fp:
    #  pickle.dump(ValidateX, fp)
    fp.write(str(ValidateX))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset-validate-Y_', 'w') as fp:
    #  pickle.dump(ValidateY, fp)
    fp.write(str(ValidateY))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset_finaltest_X', 'w') as fp:
    #pickle.dump(FinaltestX, fp)
    fp.write(str(FinaltestX))
with open('../VUDENC_data/' + 'elke_' + mode + '_dataset_finaltest_Y', 'w') as fp:
    #pickle.dump(FinaltestY, fp)
    fp.write(str(FinaltestY))
# print("saved finaltest.")

# Prepare the data for the LSTM model

#X_train = numpy.array(TrainX)
#y_train = numpy.array(TrainY)
#X_test = numpy.array(ValidateX)
#y_test = numpy.array(ValidateY)
#X_finaltest = numpy.array(FinaltestX)
#y_finaltest = numpy.array(FinaltestY)

# in the original collection of data, the 0 and 1 were used the other way round, so now they are switched so that "1" means vulnerable and "0" means clean.
# TODO: Not necessary anymore, because I changed it in VUDENC_utils.py already: line 304 and 306
"""for i in range(len(y_train)):
    if y_train[i] == 0:
        y_train[i] = 1
    else:
        y_train[i] = 0

for i in range(len(y_test)):
    if y_test[i] == 0:
        y_test[i] = 1
    else:
        y_test[i] = 0

for i in range(len(y_finaltest)):
    if y_finaltest[i] == 0:
        y_finaltest[i] = 1
    else:
        y_finaltest[i] = 0"""

"""now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
#print("numpy array done. ", nowformat)

print(str(len(X_train)) + " samples in the training set.")
print(str(len(X_test)) + " samples in the validation set.")
print(str(len(X_finaltest)) + " samples in the final test set.")

csum = 0
for a in y_train:  # TODO: y_train ist vulnerable, x_train not vulnerable or vice versa? wo wird das definiert, rausgezogen?
    csum = csum + a
print("percentage of vulnerable samples: " + str(int((csum / len(X_train)) * 10000) / 100) + "%")

testvul = 0
for y in y_test:
    if y == 1:
        testvul = testvul + 1
print("absolute amount of vulnerable samples in test set: " + str(testvul))

max_length = fulllength"""
