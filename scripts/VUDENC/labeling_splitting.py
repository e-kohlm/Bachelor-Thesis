import utils
import sys
import os.path
import json
from datetime import datetime
import random
import pickle
import numpy
from contextlib import redirect_stdout

"""
This code was first implemented in VUDENC, in the makemodel.py. 
Some changes have been made:
- the code snippets are stored differently, so they can be used with a Hugging Face model for finetuning.
  The data themselves or the method with which they are labeled has not been changed. 
- But there is no w2v vectorization made, since the transformer model itself takes care of it.
- some comments were added or rather a few changes to variables etc. if it helps for understanding
- unused code was removed, e.g. the training part for the LSTM model
"""

# default mode / type of vulnerability
mode = "sql"

# get the vulnerability from the command line argument
"""if (len(sys.argv) > 1):
    mode = sys.argv[1]"""

progress = 0
count = 0

# paramters for the filtering and creation of samples
restriction = [20000, 5, 6, 10]  # which samples to filter out
step = 5  # step lenght n in the description
fulllength = 200  # context length m in the description

# load data
with (open('../../VUDENC_data/plain_' + mode, 'r') as infile):
    data = json.load(infile)

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("finished loading. ", nowformat)


allblocks = []
snippet_id = 0
i = 0               
for repository in data:
    i += 1          # TODO: Just for the example code, the feasibility test, remove later
    if i < 4:       # TODO: Just for the example code, the feasibility test, remove later
    
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
                                pos = utils.findposition(bad, sourcecode)
                                if not -1 in pos:
                                    allbadparts.append(bad)
                        if len(allbadparts) > 0:
                            positions = utils.findpositions(allbadparts, sourcecode)

                            # get the file split up in samples
                            blocks = utils.getblocks(sourcecode, positions, step, fulllength)
                            #print("blocks: ", blocks)
                            for b in blocks:                         
                                block_dict = {}   
                                block_dict['snippet_id'] = snippet_id                                                           
                                block_dict['code'] = b[0]
                                block_dict['label'] = b[1]
                                allblocks.append(block_dict)   
                                snippet_id += 1                           

keys = []

# randomize the sample and split into train, validate and final test set
print("number of code snippets: ", len(allblocks))
for i in range(len(allblocks)):
    keys.append(i)

random.shuffle(keys)

cutoff = round(0.7 * len(keys))  # 70% for the training set
cutoff2 = round(0.85 * len(keys))  # 15% for the validation set and 15% for the final test set

keystrain = keys[:cutoff]              
keysvalidation = keys[cutoff:cutoff2]  
keystest = keys[cutoff2:] 

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

print("Creating test dataset...")
for k in keystest:
    block = allblocks[k]
    test_set.append(block)

print("Train length: " + str(len(training_set))) 
print("Validation length: " + str(len(validation_set)))
print("Testing length: " + str(len(test_set)))

now = datetime.now()
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)

# saving samples
with open('../../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset-TRAINING', 'w') as fp:
    fp.write(json.dumps(training_set))

with open('../../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset-VALIDATION', 'w') as fp:
    fp.write(json.dumps(validation_set))
    
with open('../../VUDENC_data/' + 'EXAMPLE_' + mode + '_dataset-TESTING', 'w') as fp:
    fp.write(json.dumps(test_set))
