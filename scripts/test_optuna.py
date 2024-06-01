import os
import time
import math
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import (AutoTokenizer,
                        TrainingArguments,
                         Trainer,
                          T5ForSequenceClassification,
                           pipeline,
                           DataCollatorWithPadding,
                           AutoModelForSequenceClassification,
                           AutoModelForSeq2SeqLM)
from accelerate import Accelerator
import json
import evaluate
import torch
import numpy as np
#import pandas as pd
import optuna


def run_training(args, model, train_data, tokenizer, trial=None):    
    
    start_time = time.time() 

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")    


    def compute_metrics(eval_pred):        
        predictions, labels = eval_pred 
        tuple_element_1 = np.asarray(predictions[0])
        tuple_element_2 = np.asarray(predictions[1])
        predictions = np.argmax(tuple_element_1, axis=1)
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])    
        return clf_metrics.compute(predictions=predictions, references=labels)

    # Hyperparameters to optimize
    if trial:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
        per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)
    else:
        learning_rate = args.lr
        num_train_epochs = args.epochs
        per_device_train_batch_size = args.batch_size_per_replica
        gradient_accumulation_steps = args.grad_acc_steps  


    training_args = TrainingArguments(
      
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',
        #save_strategy="no", geht nicht, denn evaluation_strategy muss gleich sein to save_strategy; 
        evaluation_strategy="epoch",            

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        #learning_rate=args.lr,
        learning_rate=learning_rate
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        #logging_steps=args.log_freq, # commented out for gpu
        #save_total_limit=1, # commented out for gpu

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
        push_to_hub=False, 
        metric_for_best_model="f1", #new
        load_best_model_at_end=True, #added later by me        

    )

    trainer = Trainer(
        model=model,
        args=training_args,        
        train_dataset=train_data["train"],
        eval_dataset=train_data["validation"],        
        tokenizer=tokenizer,        
        compute_metrics=compute_metrics,       

    )

    """best_trials=trainer.hyperparameter_search(
            direction=["minimize", "maximize"],
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=20,
            compute_objective=compute_objective,
    )"""

    
   
    trainer.train()
    trainer.save_model() # added by me later: worked with cpu, sql mit 'data_num': 5000 samples
    

    

    # Evaluate the model on the test dataset   
    finaltest_set = train_data['test']

    results = trainer.evaluate(eval_dataset=finaltest_set)  
    prediction = trainer.predict(test_dataset=finaltest_set)
    print("results: ", results)
    print("prediction: ", prediction)

    if args.local_rank in [0, -1]: 
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")         
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')
        
    
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)),"\n" )


def load_tokenize_data(args, tokenizer):      
    vulnerability_type = args.vuln_type
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
                 
            return tokenizer(examples["code"], truncation=True, max_length=tokenizer.model_max_length, padding='max_length')
      
        train_data = datasets.map(
            preprocess_function,
            #data_collator, #neu
            batched=True,            
            num_proc=64,           
        )    

        train_data = train_data.remove_columns(["snippet_id"])
        train_data = train_data.rename_column("label", "labels")
        train_data.set_format("torch")
        #print("train_data: ", train_data)         

        print(f'\n  ==> Tokenized {len(train_data)} samples')        
        train_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')
        return train_data
        
def objective(trial):
    args = get_args()  # Assuming you have a function to get the default args
    args.lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
    args.epochs = trial.suggest_int('epochs', 3, 10)
    args.batch_size_per_replica = trial.suggest_categorical('batch_size_per_replica', [8, 16, 32])
    args.grad_acc_steps = trial.suggest_int('grad_acc_steps', 1, 4)

    argsdict = vars(args) 
    #print("Arguments:\n", pprint.pformat(argsdict))
    
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))
 
    #tokenizer_max_len = 512                                        # mit order ohne das hat keine Auswirkung auf speed
    #tokenizer_config = {'max_len': tokenizer_max_len}    
    tokenizer = AutoTokenizer.from_pretrained(args.load) #, **tokenizer_config)

    train_data = load_tokenize_data(args, tokenizer=tokenizer)  

    # Check if an argument to test a smaller sample of data was given
    if args.data_num != -1:             
        train_data['train'] = [train_data['train'][i]for i in range(math.ceil(args.data_num * 70 /100))]        
        train_data['validation'] = [train_data['validation'][i]for i in range(math.ceil(args.data_num * 15 / 100))]
        train_data['test'] = [train_data['test'][i]for i in range(math.ceil(args.data_num * 15 / 100))]       
    
 
    id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}   
    label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1} 
    #accelerator = Accelerator()
    #device = accelerator.device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.load, #full xss heute nur 147 und sql nur 149 Stunden???
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id).to(device)
    #model = accelerator.prepare(model) #, training_dataloader, scheduler) #optimizer
    """for batch in training_dataloader:
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.backward(loss)
        #optimizer.step()
        #scheduler.step()"""
                                                         
    print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    f1 = run_training(args, model, train_data, tokenizer=tokenizer, trial=trial)
    return f1

def main(args): 
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    argsdict = vars(args) 
    #print("Arguments:\n", pprint.pformat(argsdict))
    
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))
 
    #tokenizer_max_len = 512                                        # mit order ohne das hat keine Auswirkung auf speed
    #tokenizer_config = {'max_len': tokenizer_max_len}    
    tokenizer = AutoTokenizer.from_pretrained(args.load) #, **tokenizer_config)

    train_data = load_tokenize_data(args, tokenizer=tokenizer)  

    # Check if an argument to test a smaller sample of data was given
    if args.data_num != -1:             
        train_data['train'] = [train_data['train'][i]for i in range(math.ceil(args.data_num * 70 /100))]        
        train_data['validation'] = [train_data['validation'][i]for i in range(math.ceil(args.data_num * 15 / 100))]
        train_data['test'] = [train_data['test'][i]for i in range(math.ceil(args.data_num * 15 / 100))]       
    
 
    id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}   
    label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1} 
    #accelerator = Accelerator()
    #device = accelerator.device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.load, #full xss heute nur 147 und sql nur 149 Stunden???
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id).to(device)
    #model = accelerator.prepare(model) #, training_dataloader, scheduler) #optimizer
    """for batch in training_dataloader:
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.backward(loss)
        #optimizer.step()
        #scheduler.step()"""
                                                         
    print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data, tokenizer=tokenizer)

def get_args():
#if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on sequence classification task")
    parser.add_argument('--vuln-type', default="sql", type=str)  
    parser.add_argument('--data-num', default=-1, type=int)  
    parser.add_argument('--cache-data', default='../cache_data', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str) 

    # Training
    parser.add_argument('--epochs', default=10, type=int) # epochs
    parser.add_argument('--lr', default=5e-5, type=float) # learning rate
    parser.add_argument('--lr-warmup-steps', default=200, type=int) # learning rate
    parser.add_argument('--batch-size-per-replica', default=1, type=int) # nicht dasselbe wie batch size, denke ich default=8
    #parser.add_argument('--batch-size', default=256, type=int)  #   nicht aus ursprünglichem fine-tuning sondern andere Stelle codeT5 
    parser.add_argument('--grad-acc-steps', default=4, type=int) # instead of updating the model parameters after processing each batch, macht also normale batch size obsolet
    parser.add_argument('--local_rank', default=-1, type=int) # irgendwas mit distributed training
    parser.add_argument('--deepspeed', default=None, type=str) # intetration with deepspeed library
    parser.add_argument('--fp16', default=False, action='store_true') # with mixed precision for training acceleration

    # Logging and stuff
    parser.add_argument('--save-dir', default="../saved_models/", type=str)
    #parser.add_argument('--log-freq', default=10, type=int)# commented out for gpu
    #parser.add_argument('--save-freq', default=500, type=int)       # default = 500 # commented out for gpu

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
    