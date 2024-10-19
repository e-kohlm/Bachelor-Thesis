import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


def load_tokenize_data(args, tokenizer):    
    vulnerability_type = args.vuln_type
    print("\n  ==> Vulnerability type: ", vulnerability_type)

    # Check if train_data already exists in cache_data/
    if os.path.exists(args.cache_data):
        train_data = load_from_disk(args.cache_data)
        print(f'\n  ==> Loaded {len(train_data)} samples')
        return train_data
    
    # Load data
    else: 
        file_path = "../VUDENC_data/"
        training_set = vulnerability_type + "_dataset-TRAINING"
        validation_set = vulnerability_type + "_dataset-VALIDATION"
        test_set = vulnerability_type + "_dataset-TESTING"
        data_files = {"train": file_path + training_set, "validation": file_path + validation_set, "test": file_path + test_set}
        datasets = load_dataset("json", data_files=data_files)

        def preprocess_function(examples):
            tokenized_examples = tokenizer(examples["code"], truncation=True, padding=True)
            return tokenized_examples

        train_data = datasets.map(
            preprocess_function,
            batched=True,
        )    

        train_data = train_data.remove_columns(["snippet_id"])
        train_data = train_data.rename_column("label", "labels")
        train_data.set_format("torch")             

        print(f'\n  ==> Tokenized {len(train_data)} samples')        
        train_data.save_to_disk(args.cache_data)
        print(f'\n  ==> Saved to {args.cache_data}')
        return train_data
