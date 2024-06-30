import os
from datasets import load_dataset, load_from_disk


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

        #data_collator = DataCollatorWithPadding(tokenizer=tokenizer) #neu

        def preprocess_function(examples):

            tokenized_examples = tokenizer(examples["code"], truncation=True, padding=True) #padding: True = pad to the longest sequence in the batch
                                                                              #truncation: True = truncate to the maximum length accepted by the model if no max_length is provided which is 512
                 
            #return tokenizer(examples["code"], truncation=True, max_length=tokenizer.model_max_length, padding='max_length')

            # Longest sequence in the batch
            """max_length = max(len(seq) for seq in tokenized_examples['input_ids']) #new
            print(f"Max sequence length in batch: {max_length}")

            print("\nDetailed info about tokenized examples:")
            print(f"Number of examples: {len(tokenized_examples['input_ids'])}")
            for i in range(min(3, len(tokenized_examples['input_ids']))):
                print(f"\nExample {i+1}:")
                print(f"  Input IDs: {tokenized_examples['input_ids'][i]}")
                print(f"  Attention Mask: {tokenized_examples['attention_mask'][i]}")
                if 'token_type_ids' in tokenized_examples:
                    print(f"  Token Type IDs: {tokenized_examples['token_type_ids'][i]}")
                print(f"  Length: {len(tokenized_examples['input_ids'][i])}")
            
            print("\nFull structure of tokenized examples:")
            for key, value in tokenized_examples.items():
                print(f"{key}: {value[:3]}")"""
            return tokenized_examples

        train_data = datasets.map(
            preprocess_function,
            #data_collator, #neu
            batched=True,            
            #num_proc=16,           
        )    

        train_data = train_data.remove_columns(["snippet_id"])
        train_data = train_data.rename_column("label", "labels")
        train_data.set_format("torch")             

        print(f'\n  ==> Tokenized {len(train_data)} samples')        
        train_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')
        return train_data
