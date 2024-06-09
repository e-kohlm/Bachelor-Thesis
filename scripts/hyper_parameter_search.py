import os
import time
import math
import pprint
import argparse
from fine_tuning import load_tokenize_data
from datasets import load_dataset, load_from_disk
from transformers import (
                        AutoTokenizer,
                        TrainingArguments,
                        Trainer, 
                        AutoModelForSequenceClassification,
                        EarlyStoppingCallback
                        )
import json
import evaluate
import torch
import numpy as np
import optuna


def run_training(args, train_data, tokenizer, trial=None):    
    print("\n run training trial: ", trial)   
    
    start_time = time.time()

    def model_init(trial):        
        id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}   
        label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1}    
        device = "cpu"  # "cuda" for gpu       
        model = AutoModelForSequenceClassification.from_pretrained(args.load,
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id).to(device)                                                      
        print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")
        return model 


    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_metrics(eval_pred):        
        print("\n CCCCcompute_metrics: ")
        predictions, labels = eval_pred 
        tuple_element_1 = np.asarray(predictions[0])
        tuple_element_2 = np.asarray(predictions[1])
        predictions = np.argmax(tuple_element_1, axis=1)
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])    
        return clf_metrics.compute(predictions=predictions, references=labels)

    # todo die alle überprüfen, wenn sie keinen Sinn machen zu testen, dann auch wieder zu args und default werte
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
            #"batch_size_per_replica": trial.suggest_categorical('batch_size_per_replica', [8, 16, 32]), 
            "gradient_accumulation_steps": trial.suggest_int('gradient_accumulation_steps', 1, 4),    
            "warmup_steps": trial.suggest_int('warmup_steps', 100, 200)        
        }  

    training_args = TrainingArguments(                
        report_to='tensorboard',        
        do_train=True,
        save_strategy='epoch',        
        eval_strategy="epoch",
        num_train_epochs=args.epochs,
        #metric_for_best_model="f1",
        #load_best_model_at_end=True, 
        
        learning_rate=optuna_hp_space(trial)["learning_rate"],
        per_device_train_batch_size=optuna_hp_space(trial)["per_device_train_batch_size"],        
        gradient_accumulation_steps=optuna_hp_space(trial)["gradient_accumulation_steps"],     
        weight_decay=0.05,
        warmup_steps=optuna_hp_space(trial)["warmup_steps"],

        output_dir=args.save_dir,
        overwrite_output_dir=False,
        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq, # commented out for gpu
        save_total_limit=1, # commented out for gpu
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
        push_to_hub=False,        
        )  

    trainer = Trainer(
        model = None,
        args=training_args,        
        train_dataset=train_data["train"],
        eval_dataset=train_data["validation"],        
        tokenizer=tokenizer,        
        compute_metrics=compute_metrics,    
        model_init=model_init 
        # early_stopping_patience: Use with metric_for_best_model to stop training when the specified metric worsens for early_stopping_patience evaluation calls
        # early_stopping_threshold: Use with TrainingArguments metric_for_best_model and early_stopping_patience to denote how much the specified metric must improve to satisfy early stopping conditions. 
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01)]        

    )   
    
    print(f'\n  ==> Started trial {trial.number}')
    best_trial = trainer.hyperparameter_search(                    
        direction="maximize",
        backend="optuna",
        n_trials=args.n_trials,
        hp_space=optuna_hp_space,
        #compute_objective=compute_objective,
    ) 
    print("\nBest trial:\n", best_trial)

    end_time = time.time()
    time_elapsed = end_time - start_time    
    print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)),"\n" )
    
    return trainer.evaluate()['eval_f1']

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
    

    def objective(trial):
        xxx = run_training(args=args, train_data=train_data, tokenizer=tokenizer, trial=trial)            
        # You could define your own compute_objective function, if not defined, the default compute_objective will be called,
        # and the sum of eval metric like f1 is returned as objective value.
        #return run_training(args=args, train_data=train_data, tokenizer=tokenizer, trial=trial)  
        # return evaluation_score ist richtig 
        print("TEST f1 value ist das : ", xxx)
        #results = trainer.evaluate(eval_dataset=finaltest_set)    
        

       
        
    
    study = optuna.create_study(
        #study_name='hp_search_' + args.vuln_type,
        direction="maximize",
        storage='sqlite:///hp_search.db',
        #load_if_exists=True
        )
    test = study.optimize(objective, n_trials=args.n_trials)
    print("test: ", test)
    print("Best trial:")
    #trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")    


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on sequence classification task")
    parser.add_argument('--vuln-type', default="sql", type=str)  
    parser.add_argument('--data-num', default=-1, type=int)  
    parser.add_argument('--cache-data', default='../cache_data', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str) 

    # Hyperparameter search    
    parser.add_argument('--n_trials', default=10, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int) # nicht dasselbe wie batch size, denke ich default=8        
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
