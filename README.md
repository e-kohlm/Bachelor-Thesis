## Preprocessing VUDENC Data
Download of VUDENC datasets from Zenodo:   
https://zenodo.org/record/3559841
* plain_command_injection   
* plain_open_redirect   
* plain_path_disclosure   
* plain_remote_code_execution  
* plain_sql   
* plain_xsrf   
* plain_xss   


The files need to be stored in the directory VUDENC_data for the script to run successfully.   
Labeling and splitting of the data is done with labeling_splitting.py (which imports some functions from utils.py).   
Run the script from `scripts/VUDENC/` with:    

`python labeling_splitting.py <mode>` 

Possible modes are command_injection, sql, and so on.   
The results are stored in VUDENC_data, e.g.:  

* sql_dataset-TESTING  
* sql_dataset-TRAINING  
* sql_dataset-VALIDATION    

accordingly for the other six vulnerability types. There are smaller datasets with less code samples (three files with prefix EXAMPLE) provided.   
Each file contains code snippets with a label (0 = not vulnerable, 1 = vulnerable) for each.

# Training
Scripts for hyperparameter search and finetuning need to run from `scripts/`. All training scripts can be called with different arguments, you can find them at the end of each script.  
Default device is a GPU, but it can be run on a CPU as well, with the argument `--device`.   
The default model is  Salesforce/codet5p-220m, which can be changed with the argument `--load`.   
Default vulnerability type is sql, it can be changed with `--vuln_type`.

## Hyperparameter Search
The arguments with which finetuning.py can be called are specified in the file.   
To find the optimal hyperparameters for the models run

`python hyperparameters_search.py`   



## Finetuning


The arguments with which finetuning.py can be called are specified in the file.   

E.g. for testing it can determine the number of epochs and number of samples like this:  

`python finetuning.py --epochs=2 --data-num=50`  

The number of samples used should not be smaller than 50, if it is, it crashes along the way. 

## Inference 
To test the finetuned model the script `inference.py` can be run. It gives an idea how to use the model. Since the finetuned model is not made publicly available, the training must be made before it is possible to test.  
The result of an example prediction is saved as `predictions/predictions_EXAMPLE_sql.txt`.

