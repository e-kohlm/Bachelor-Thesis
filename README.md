# Bachelor-Thesis
## Description of the Feasibility Study
Still draft version

## Step 1
Downloaded VUDENC dataset plain_sql from Zenodo: URL https://zenodo.org/record/3559841#.XeVaZNVG2Hs, it is stored in VUDENC_data.
Labeling and splitting of the data is done with labeling_splitting.py (which imports some functions from utils.py).
Run the script with:  

`python labeling_splitting.py`  

The results are  three files:  

* EXAMPLE_sql_dataset-TESTING  
* EXAMPLE_sql_dataset-TRAINING  
* EXAMPLE_sql_dataset-VALIDATION    

which are code snippets with a label for each, stored in VUDENC_data.  

## Step ?
The script `fine_tuning.py` can be called with certain arguments, specified in the file.
E.g. for testing it can determine the number of epochs and number of samples like this:
`python fine_tuning.py --epochs=1 --data-num=50`  
The number of samples used should not be to small. 

Default is the 220m model, but 770m auch mit Argumenten
