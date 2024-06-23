import os
import sys
import time
import math
import pprint
import argparse
from gpu_fine_tuning import load_tokenize_data
from datasets import load_dataset, load_from_disk
from transformers import (
                        AutoTokenizer,
                        TrainingArguments,
                        Trainer, 
                        AutoModelForSequenceClassification                        
                        )
import json
import evaluate
import torch
import numpy as np
import optuna
import logging
#from mpi4py import MPI
#os.environ['CUDA_LAUNCH_BLOCKING']="1"


def optuna_hp_space(trial):
    print("trial optuna_hp_space: ", trial)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "batch-size-per-replica": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),            
        "gradient_accumulation_steps": trial.suggest_int('gradient_accumulation_steps', 1, 4),    
        "warmup_steps": trial.suggest_int('warmup_steps', 100, 500),
        "epochs": trial.suggest_int('epochs', 1, 10),   
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 0.1),    # kein model-parameter, prevents over or underfitting
        #"dropout_rate": trial.suggest_uniform("dropout_rate", 0.05, 0.5, , log=True) # is a model parameter
        #"optimizer": trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd", "adagrad"]),            
    } 

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

def objective(trial, train_data, tokenizer, args):
    print("trial objective: ", trial)
    print(f'\n  ==> Started trial {trial.number}')  

    def model_init(trial):        
        id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}   
        label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1}    
        device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage     
        model = AutoModelForSequenceClassification.from_pretrained(args.load,
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id).to(device)
        print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")                                                         
        return model

    training_args = TrainingArguments(                
        report_to='tensorboard', 
        output_dir=args.save_dir,
        overwrite_output_dir=False,       
        
        #do_train=True,
        save_strategy='epoch',        
        eval_strategy="epoch",        
        metric_for_best_model="f1",
        load_best_model_at_end=True, 
        save_total_limit=1, 
        
        num_train_epochs=optuna_hp_space(trial)["epochs"],
        per_device_train_batch_size=optuna_hp_space(trial)["batch-size-per-replica"],        
        gradient_accumulation_steps=optuna_hp_space(trial)["gradient_accumulation_steps"],             
        learning_rate=optuna_hp_space(trial)["learning_rate"],
        weight_decay=optuna_hp_space(trial)["weight_decay"], # model parameter default CodeT5+ = 0.05
        warmup_steps=optuna_hp_space(trial)["warmup_steps"],               
   
        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq, 
        
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,        
        )  

    trainer = Trainer(
        model = None,
        args=training_args,        
        train_dataset=train_data["train"],
        eval_dataset=train_data["validation"],        
        tokenizer=tokenizer,        
        compute_metrics=compute_metrics,    
        model_init=model_init
    )  

    # Train the model
    trainer.train()    

    # Evaluate the model and return the desired metric
    eval_result = trainer.evaluate()
    return eval_result['eval_f1']     


def main(args): 
    argsdict = vars(args) 
    #print("Arguments:\n", pprint.pformat(argsdict))
   
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))


    tokenizer = AutoTokenizer.from_pretrained(args.load)
    train_data = load_tokenize_data(args, tokenizer=tokenizer) 
    print("train_data:\n", train_data)
    
    # For training a smaller sample, if args != -1 is given
    if args.data_num != -1:             
        train_data['train'] = [train_data['train'][i]for i in range(math.ceil(args.data_num * 70 /100))]        
        train_data['validation'] = [train_data['validation'][i]for i in range(math.ceil(args.data_num * 15 / 100))]
        train_data['test'] = [train_data['test'][i]for i in range(math.ceil(args.data_num * 15 / 100))] 
        
    start_time = time.time()
    def wrapped_objective(trial):
        print("trial wrapped_objective: ", trial)
        return objective(trial, train_data=train_data, tokenizer=tokenizer, args=args)
    
    #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        direction="maximize",
        storage='sqlite:///../hp_search/hp_search.db',      
        #compute_objective=wrapped_objective,
        #sampler=optuna.samplers.RandomSampler(),
        #pruner=optuna.pruners.MedianPruner(),
        #load_if_exists=True        
    )
    print("study: ", study)
    #print("study.name: ", study.name)
    print("rapped_obj: ", wrapped_objective)


    

    # Run optimization   
    study.optimize(wrapped_objective, n_trials=args.n_trials) 


    # Retrieve best trial
    best_trial = study.best_trial
    #print("\nBest trial: \n", best_trial.params)
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")   

    end_time = time.time()
    time_elapsed = end_time - start_time    
    print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)),"\n" )

# Optionally, load the best model and re-train or evaluate further
#best_params = best_trial.params
#model_name = 'Salesforce/codet5p-220m'
#model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=2)
#tokenizer = T5Tokenizer.from_pretrained(model_name)
#train_dataset, eval_dataset = load_my_data()



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on sequence classification task")
    parser.add_argument('--vuln-type', default="sql", type=str)  
    parser.add_argument('--data-num', default=-1, type=int)  
    parser.add_argument('--cache-data', default='../cache_data', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str) 

    # Hyperparameter search    
    parser.add_argument('--n_trials', default=10, type=int)    

    # Speeding up          
    parser.add_argument('--local_rank', default=-1, type=int) # irgendwas mit distributed training
    parser.add_argument('--deepspeed', default=None, type=str) # intetration with deepspeed library
    parser.add_argument('--fp16', default=False, action='store_true') # with mixed precision for training acceleration

    # Logging and stuff
    parser.add_argument('--save-dir', default="../hp_search/", type=str)
    parser.add_argument('--log-freq', default=10, type=int)# commented out for gpu
    parser.add_argument('--save-freq', default=500, type=int)       # default = 500 # commented out for gpu   

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
