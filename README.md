## Preprocessing VUDENC Data
Download of VUDENC datasets from Zenodo:   
https://zenodo.org/record/3559841#XeVaZNVG2Hs
* plain_command_injection   
* plain_open_redirect   
* plain_path_disclosure   
* plain_remote_code_execution  
* plain_sql   
* plain_xsrf   
* plain_xss   


The files need to be stored in the directory VUDENC_data for the script to run successfully.   
Labeling and splitting of the data is done with¸ labeling_splitting.py (which imports some functions from `utils.py`).
Run the script from `scripts/VUDENC/` with:  

`python labeling_splitting.py <mode>` 

Possible modes are command_injection, sql, and so on.   
The results are stored in VUDENC_data:  

* sql_dataset-TESTING  
* sql_dataset-TRAINING  
* sql_dataset-VALIDATION    

and for the other six vulnerabilities accordingly. There are smaller datasets with less code samples (three files with prefix EXAMPLE) provided. The files can be downloaded from
https://zenodo.org/uploads/10962553   

ACHTUNG: Ist noch nicht die richtige url , so lange es noch nicht veröffentlich wurde.

Each file contains code snippets with a label (0 = not vulnerable, 1 = vulnerable) for each.



## Finetuning
Run the script from `scripts/`.   
On a GPU the job is run with Slurm, so the arguments are specified in the script `pytorch_gpu.sh`.   
To start a job on a GPU use the command `sbatch pytorch_gpu.sh`.   

If a CPU is used the script `finetuning.py` needs to be called directly and the default device needs to be changed by passing the argument `--device=cpu`. 
The arguments with which finetuning.py can be called are specified in the file.

E.g. for testing it can determine the number of epochs and number of samples like this:  

`python finetuning.py --epochs=2 --data-num=50`  

The number of samples used should not be smaller than 50, if it is, it crashes along the way. 

Default for now is the model CodeT5+ 220m

The training with the EXAMPLE_sql files of full size had a runtime of around 20 hours on  gruenau 3. This very, very much time was hopefully due to the reduced service because of the air-conditioning issues of the institute.

## Step 3
 
Theoretically, to test the fine-tuned model the script `inference.py` can be run. It gives an idea how to use the model. Since the model hasn't been pushed to git, for now, the training must be made before it is possible to test.  
The result of an example prediction is saved as `predictions/predictions_EXAMPLE_sql.txt`.

