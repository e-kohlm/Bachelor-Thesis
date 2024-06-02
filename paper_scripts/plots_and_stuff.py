import pprint
import argparse
from datasets import load_dataset
import pandas as pd

def load_data(vuln):
    file_path = "../VUDENC_data/"
    training_set = vuln + "_dataset-TRAINING"
    validation_set = vuln + "_dataset-VALIDATION"
    test_set = vuln + "_dataset-TESTING"

    data_files = {"train": file_path + training_set, "validation": file_path + validation_set, "test": file_path + test_set}
    datasets = load_dataset("json", data_files=data_files)   
    print(datasets)
    return datasets

def get_label_balance(dataset):        
    df_dataset = pd.DataFrame(dataset)
    df_dataset.set_index('snippet_id')   
    print("data types:", df_dataset.dtypes)
    print("size:", df_dataset.size)
    print("shape:", df_dataset.shape)
    print("rows: ", df_dataset.shape[0])
    print("columns: ", df_dataset.shape[1])
    print("head:", df_dataset.head(10))
    number_not_vulnerable = (df_dataset['label'].values == 0).sum()
    number_vulnerable = (df_dataset['label'].values == 1).sum()
    print("number not vuln: ", number_not_vulnerable)
    print("number vuln: ", number_vulnerable)   
   
    
    test = number_vulnerable + number_not_vulnerable   
    if test != df_dataset.shape[0]:
        print("UWAGA!")    

    balance_dict = {}
    balance_dict['number_rows'] = df_dataset.shape[0]
    balance_dict['number_vulnerable'] = number_vulnerable 
    balance_dict['number_not_vulnerable'] = number_not_vulnerable 

    return balance_dict

def compute_percent_vulnerable(train_balance):
    percentage_vulnerable = (100 * train_balance['number_vulnerable']) / train_balance['number_rows'] 
    return percentage_vulnerable


def main(args): 
    argsdict = vars(args) 
    print("Arguments:\n", pprint.pformat(argsdict))
    datasets = load_data(argsdict["vuln"])   
    train_balance = get_label_balance(datasets['train'])
    print("train_balance: ", train_balance)
    percentage_vulnerable = compute_percent_vulnerable(train_balance)
    print("percent_vuln: ", percentage_vulnerable)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Get to know the data and stuff")
    parser.add_argument('--vuln', default="sql", type=str)  

    args = parser.parse_args()
    #os.makedirs(args.save_dir, exist_ok=True)
   
    main(args)