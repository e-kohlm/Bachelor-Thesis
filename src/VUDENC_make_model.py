import VUDENC_utils
import sys
import os.path
import json
from datetime import datetime
import random
import pickle
#import numpy
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.preprocessing import sequence
#from keras import backend as K
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score
#from sklearn.utils import class_weight
#import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors

# TODO: von VUDENC geklaut, insofern sollt ich das villeicht auch in eine extra dir: VUDENC_src packen

"""
run from /src with command python VUDENC_make_model.py sql
Running without error with sql as example
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

# TODO: w2v Zeug brauche ich nicht
"""### paramters for the filtering and creation of samples
restriction = [20000, 5, 6, 10]  # which samples to filter out
step = 5  # step lenght n in the description
fulllength = 200  # context length m in the description

mode2 = str(step) + "_" + str(fulllength)

### hyperparameters for the w2v model
mincount = 10  # minimum times a word has to appear in the corpus to be in the word2vec model
iterationen = 100  # training iterations for the word2vec model
s = 200  # dimensions of the word2vec model
w = "withString"  # word2vec model is not replacing strings but keeping them

# get word2vec model
w2v = "word2vec_" + w + str(mincount) + "-" + str(iterationen) + "-" + str(s)
w2vmodel = "w2v/" + w2v + ".model"  # das ist das bereits erstellte model mit den o.g. parametern, also die Datei:
# word2vec_withString-10-100-200.model
# hier im Project habe ich das von mir trainierte w2v, falls ich neu trainieren muss
# zum testen reicht aber vielleicht auch das alte (vom 10.12.????

# load word2vec model
if not (os.path.isfile(w2vmodel)):
    print("word2vec model is still being created...")
    sys.exit()

w2v_model = Word2Vec.load(w2vmodel)
word_vectors = w2v_model.wv"""


# load data
with (open('../VUDENC_data/plain_' + mode, 'r') as
      infile):  # Input if mode=sql, dann der File plain_sql from Code/data usw.
    data = json.load(infile)

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("finished loading. ", nowformat)

### meins ###### filtering of unwanted code, I think
allblocks = []

for r in data:  # repository
    progress = progress + 1

    for c in data[r]:  # commit

        if "files" in data[r][c]:
            #  if len(data[r][c]["files"]) > restriction[3]:
            # too many files
            #    continue

            for f in data[r][c]["files"]:

                #      if len(data[r][c]["files"][f]["changes"]) >= restriction[2]:
                # too many changes in a single file
                #       continue

                if not "source" in data[r][c]["files"][f]:
                    # no sourcecode
                    continue

                if "source" in data[r][c]["files"][f]:
                    sourcecode = data[r][c]["files"][f]["source"]
                    #     if len(sourcecode) > restriction[0]:
                    # sourcecode is too long
                    #       continue

                    allbadparts = []

                    for change in data[r][c]["files"][f]["changes"]:

                        # get the modified or removed parts from each change that happened in the commit
                        badparts = change["badparts"]
                        count = count + len(badparts)

                        #     if len(badparts) > restriction[1]:
                        # too many modifications in one change
                        #       break

                        for bad in badparts:
                            # check if they can be found within the file
                            pos = VUDENC_utils.findposition(bad, sourcecode)
                            if not -1 in pos:
                                allbadparts.append(bad)

                    #   if (len(allbadparts) > restriction[2]):
                    # too many bad positions in the file
                    #     break

                    if (len(allbadparts) > 0):
                        #   if len(allbadparts) < restriction[2]:
                        # find the positions of all modified parts
                        positions = VUDENC_utils.findpositions(allbadparts, sourcecode)

                        # get the file split up in samples
                        blocks = VUDENC_utils.getblocks(sourcecode, positions, step, fulllength)

                        for b in blocks:
                            # each is a tuple of code and label
                            allblocks.append(b)
############## meins######
##### allblocks [] ist der komplett code nach filtering
keys = []

# randomize the sample and split into train, validate and final test set
for i in range(len(allblocks)):
    keys.append(i)
random.shuffle(keys)

cutoff = round(0.7 * len(keys))  # 70% for the training set
cutoff2 = round(0.85 * len(keys))  # 15% for the validation set and 15% for the final test set

keystrain = keys[:cutoff]  # Keys der Trainingsdaten
keystest = keys[cutoff:cutoff2]  # Keys der Testdaten
keysfinaltest = keys[cutoff2:]  # Keys des final test

print("cutoff " + str(cutoff))
print("cutoff2 " + str(cutoff2))

with open('../VUDENC_data/' + mode + '_dataset_keystrain',
          'wb') as fp:  # The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream,
    pickle.dump(keystrain, fp)  # pickle module not considered secure anymore
with open('../VUDENC_data/' + mode + '_dataset_keystest', 'wb') as fp:
    pickle.dump(keystest, fp)
with open('../VUDENC_data/' + mode + '_dataset_keysfinaltest', 'wb') as fp:
    pickle.dump(keysfinaltest, fp)

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
  code = block[0]
  token = VUDENC_utils.getTokens(code) #get all single tokens from the snippet of code
  print("token = ", token)
  vectorlist = []

  """for t in token: #convert all tokens into their word2vec vector representation  # TODO: Möchte ich das???
    if t in word_vectors.vocab and t != " ":
      vector = w2v_model[t]
      vectorlist.append(vector.tolist())"""
  TrainX.append(vectorlist) #append the list of vectors to the X (independent variable)
  TrainY.append(block[1]) #append the label to the Y (dependent variable)

  print("TrainY: ", TrainY)  # Elke war das

print("Creating validation dataset...")
for k in keystest:
  block = allblocks[k]
  code = block[0]
  token = VUDENC_utils.getTokens(code) #get all single tokens from the snippet of code
  vectorlist = []
  """for t in token: #convert all tokens into their word2vec vector representation  # TODO: Möchte ich das???
    if t in word_vectors.vocab and t != " ":
      vector = w2v_model[t]
      vectorlist.append(vector.tolist())
  ValidateX.append(vectorlist) #append the list of vectors to the X (independent variable)"""
  ValidateY.append(block[1]) #append the label to the Y (dependent variable)

print("Creating finaltest dataset...")
for k in keysfinaltest:
  block = allblocks[k]
  code = block[0]
  token = VUDENC_utils.getTokens(code) #get all single tokens from the snippet of code
  vectorlist = []
  """for t in token: #convert all tokens into their word2vec vector representation  # TODO: Möchte ich das???
    if t in word_vectors.vocab and t != " ":
      vector = w2v_model[t]
      vectorlist.append(vector.tolist())
  FinaltestX.append(vectorlist) #append the list of vectors to the X (independent variable)"""
  FinaltestY.append(block[1]) #append the label to the Y (dependent variable)

print("Train length: " + str(len(TrainX)))
print("Test length: " + str(len(ValidateX)))
print("Finaltesting length: " + str(len(FinaltestX)))
now = datetime.now() # current date and time
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)
######################################################

# saving samples

#with open('data/plain_' + mode + '_dataset-train-X_'+w2v + "__" + mode2, 'wb') as fp:
#  pickle.dump(TrainX, fp)
#with open('data/plain_' + mode + '_dataset-train-Y_'+w2v + "__" + mode2, 'wb') as fp:
#  pickle.dump(TrainY, fp)
#with open('data/plain_' + mode + '_dataset-validate-X_'+w2v + "__" + mode2, 'wb') as fp:
#  pickle.dump(ValidateX, fp)
#with open('data/plain_' + mode + '_dataset-validate-Y_'+w2v + "__" + mode2, 'wb') as fp:
#  pickle.dump(ValidateY, fp)
with open('../VUDENC_data/' + mode + '_dataset_finaltest_X', 'wb') as fp:
  pickle.dump(FinaltestX, fp)
with open('../VUDENC_data/' + mode + '_dataset_finaltest_Y', 'wb') as fp:
  pickle.dump(FinaltestY, fp)
#print("saved finaltest.")