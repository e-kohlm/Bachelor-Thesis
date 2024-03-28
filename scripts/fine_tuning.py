import os
import time
import math
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, pipeline, DataCollatorWithPadding
import json
from contextlib import redirect_stdout # weg
import evaluate
import torch
import numpy as np


def run_training(args, model, train_data, tokenizer):
    print("\nStarting training")
    #print("train_data: ", train_data)  # hier sehe ich die je 512 token, was viel zu viel ist, 256 wäre immer noch reichlich, sogar 128
    # Wenn es das ist, nämlich vector dimensionality, dann hier lesen: Wartschinski-Noller et al S. 11!!!
    start_time = time.time()

    # ValueError: expected sequence of length 62 at dim 1 (got 57)

    # Setze ich padding = True: ValueError: expected sequence of length 76 at dim 1 (got 78)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")  
    # FIXME: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 due to no true samples.
    #  Use `zero_division` parameter to control this behavior.


    def compute_metrics(eval_pred):
        #print("eval_pred: ", eval_pred)
        predictions, labels = eval_pred
        #print("predictions: ", predictions)
        #print("labels: ", labels)

        # FIXME: predictions is a tuple with two elements.
        # - The first element is an array of e.g. (--data-num=80)
        # 8 tuples with two elements each => (8,2)
        # - The second has three dimensions (8, 512, 768), what to do with that
        
        #print("pred type: ", type(predictions))
        #print("lables type: ", type(labels))
        #print("test tuple element 1: ", predictions[0][0])
        #print("test tuple element 2: ", predictions[1][0])  # What exactly do numbers of the second tuple mean?

        
        tuple_element_1 = np.asarray(predictions[0])
        tuple_element_2 = np.asarray(predictions[1])        
        #print("tuple_element_1 shape: ", tuple_element_1.shape)
        print("tuple_element_2 shape: ", tuple_element_2.shape)
        #print("labels shape: ", labels.shape)

        predictions = np.argmax(tuple_element_1, axis=1)  # FIXME only prediction with values of first tuple, what to do with the other one????
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])    
        return clf_metrics.compute(predictions=predictions, references=labels)  # TODO Save to file


    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',
        evaluation_strategy="epoch",

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
        push_to_hub=False,   # really???
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        #train_dataset=train_data,
        train_dataset=train_data["train"],  # TODO train-validation-test? Or what turorial says train-validation? Or what codet5 says: train?
        eval_dataset=train_data["test"],
        tokenizer=tokenizer,
        #data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')
    
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)),"\n" )


def load_tokenize_data(args, tokenizer):  # 5.   

    # Check if train_data already exists in cache_data/summarize_python
    if os.path.exists(args.cache_data):
        train_data = load_from_disk(args.cache_data)
        print(f'\n  ==> Loaded {len(train_data)} samples')
        return train_data
    # Load data
    else:        
        file_path = "../VUDENC_data/"
        training_set = "EXAMPLE_sql_dataset-TRAINING"
        validation_set = "EXAMPLE_sql_dataset-VALIDATION"
        test_set = "EXAMPLE_sql_dataset-TESTING"
        data_files = {"train": file_path + training_set, "validation": file_path + validation_set, "test": file_path + test_set}
        datasets = load_dataset("json", data_files=data_files)        
        with open('../test_outputs/test_load_fine_tuning.txt', 'w') as f:
                with redirect_stdout(f):
                    print("datasets test: ", datasets['train'][0], "\n" )
        #print("datasets: ", datasets)   




        # Tokenize data
        """tokenizer_max_len = 512
        tokenizer_config = {'max_len': tokenizer_max_len}
        #tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
        tokenizer = AutoTokenizer.from_pretrained(args.load, **tokenizer_config)
        #print("tokenizer model_max_length: ", tokenizer.model_max_length)"""

        def preprocess_function(examples):
            with open('../test_outputs/test_fine_tuning_examples.txt', 'w') as f:
                with redirect_stdout(f):
                    print("examples 0: ", examples["code"][0])
                    #print("examples 3: ", examples["code"][3])
                    #print("examples 10: ", examples["code"][10])    
            return tokenizer(examples["code"], truncation=True, padding='max_length', max_length=tokenizer.model_max_length)
        # return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length
        # Preprocess data
        train_data = datasets.map(
            preprocess_function,
            batched=True,
            #remove_columns=datasets.column_names,
            num_proc=64,
            #load_from_cache_file=False,
        )

        train_data = train_data.remove_columns(["snippet_id"])
        train_data = train_data.rename_column("label", "labels")
        train_data.set_format("torch")
        print("train_data: ", train_data)    
        #print("train_data['train'][0]: ", train_data['train'][0], "\n" )



        print(f'\n  ==> Tokenized {len(train_data)} samples')
        #data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')
        return train_data


def main(args):  # 2.     
    argsdict = vars(args)  # args to dictinary
    print("Arguments:\n", pprint.pformat(argsdict))  # print args pretty

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:  # 3. Argumente werden in file gespeichert
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    tokenizer_max_len = 512
    tokenizer_config = {'max_len': tokenizer_max_len}
    #tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
    tokenizer = AutoTokenizer.from_pretrained(args.load, **tokenizer_config)
    #print("tokenizer model_max_length: ", tokenizer.model_max_length)
        


    train_data = load_tokenize_data(args, tokenizer=tokenizer)  # 4. Daten werden geladen und tokenized
    
    # Check if an argument for a smaller sample of data was given
    if args.data_num != -1:        
        train_data['train'] = [train_data['train'][i]for i in range(args.data_num)]
        #print("DATA: ", [train_data['train'][i]for i in range(args.data_num)])
        train_data['validation'] = [train_data['validation'][i]for i in range(math.ceil(args.data_num * 15 / 100))]
        train_data['test'] = [train_data['test'][i]for i in range(math.ceil(args.data_num * 15 / 100))]


    
    # Load model from `args.load`
    
    # einfach so auf 4 label setzen geht nicht: ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
    id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}
    #id2label ={0: "NOT VULNERABLE", 1: "SQL_VULNERABLE", 2: "XSRF_VULNERABLE", 3: "OPEN_REDIRECT_VULNERABLE"}
    label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1}
    #label2id = {"NOT VULNERABLE": 0, "SQL_VULNERABLE": 1, "XSRF_VULNERABLE": 2, "OPEN_REDIRECT_VULNERABLE": 3}
    model = AutoModelForSequenceClassification.from_pretrained(args.load, 
                                                            #num_labels=2,
                                                            num_labels=2,
                                                            id2label=id2label,
                                                            label2id=label2id)    
                                                            
    print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data, tokenizer=tokenizer)   


if __name__ == "__main__":  # Argumente 
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on sequence classification task")
    parser.add_argument('--data-num', default=-1, type=int)
    #parser.add_argument('--max-source-len', default=320, type=int)
    #parser.add_argument('--max-target-len', default=128, type=int)
    parser.add_argument('--cache-data', default='cache_data/summarize_python', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)  # TODO: auch mit codet5p-770m trainieren

    # Training
    parser.add_argument('--epochs', default=10, type=int)  # Number of times data set has to be worked through
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    #parser.add_argument('--accuracy', default=True, type=float)# meins, wird einfach ignoriert
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/summarize_python", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()  # Argumente werden geparsed, ohne die, kann main nicht aufgerufen werden

    os.makedirs(args.save_dir, exist_ok=True)  # Make directory save_dir aus args = saved_models/summarize_python

    main(args)      # 1. Start
