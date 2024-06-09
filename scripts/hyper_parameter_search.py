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


def run_training(args, train_data, tokenizer, model=None, trial=None):    
    print("\n model: ", model)
    print("\n trial: ", trial)
    
    start_time = time.time() 

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

    # Set up training arguments with hyperparameters from Optuna trial if available
    #learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True) if trial else args.lr
    #per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 16, 64, log=True) if trial else args.batch_size_per_replica

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
            #"batch_size_per_replica": trial.suggest_categorical('batch_size_per_replica', [8, 16, 32]), 
            #"grad_acc_steps": trial.suggest_int('grad_acc_steps', 1, 4)

            # Meaning, I think: Set it in Training Args
            #Trying to set batch_size_per_replica in the hyperparameter search but there is no corresponding field in `TrainingArguments`.
            #Trying to set grad_acc_steps in the hyperparameter search but there is no corresponding field in `TrainingArguments`.

        }

    if trial: 
        print("2. if trial: ", trial)
        hyperparameters = optuna_hp_space(trial)
        learning_rate = hyperparameters["learning_rate"]
        per_device_train_batch_size = hyperparameters["per_device_train_batch_size"]        
        #grad_acc_steps = hyperparameters["grad_acc_steps"] 
        epochs = 1
    else:
        learning_rate = args.lr
        per_device_train_batch_size = args.batch-size-per-replica   
        grad_acc_steps = args.grad-acc-steps
        epochs = args.epochs

    training_args = TrainingArguments(      
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,
        do_train=True,
        save_strategy='epoch',        
        eval_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=learning_rate, #commented out for hyper-param search
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,
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
        metric_for_best_model="f1",
        #load_best_model_at_end=True, 
        )      

    def model_init(trial):
        print("\n model_init trial: ", trial)
        id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}   
        label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1}    
        device = "cpu"  # "cuda" for gpu       
        model = AutoModelForSequenceClassification.from_pretrained(args.load,
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id).to(device)
        print("\n2. loaded model")                                                
        print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")
        return model


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

    if trial:  # If we're running hyperparameter search
        print("\n if trial: ", trial)
        print(f'  ==> Started trial {trial.number}')
        
        best_trial = trainer.hyperparameter_search(
                     
            direction="maximize",
            backend="optuna",
            n_trials=args.n_trials,
            hp_space=optuna_hp_space,
            #compute_objective=compute_objective,
        )

 
        print("best_trial", best_trial)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)),"\n" )
        
    else:
        print("\n if not trial: ", trial)
    
        trainer.train()    
        trainer.save_model() 

        # Evaluate the model on the test dataset   
        finaltest_set = train_data['test']

        results = trainer.evaluate(eval_dataset=finaltest_set)  
        prediction = trainer.predict(test_dataset=finaltest_set)
        print("results: ", results)
        print("prediction: ", prediction)

        if args.local_rank in [0, -1]: 
            final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")         
            model.save_pretrained(final_checkpoint_dir)
            print(f'  ==> Finished training and saved to {final_checkpoint_dir}')       
        
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)),"\n" )

def main(args): 
    argsdict = vars(args) 
    #print("Arguments:\n", pprint.pformat(argsdict))
   
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict)) 
        
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    train_data = load_tokenize_data(args, tokenizer=tokenizer) 
    print("train_data:\n ", train_data)
    
    # For training a smaller sample, if args != -1 is given
    if args.data_num != -1:             
        train_data['train'] = [train_data['train'][i]for i in range(math.ceil(args.data_num * 70 /100))]        
        train_data['validation'] = [train_data['validation'][i]for i in range(math.ceil(args.data_num * 15 / 100))]
        train_data['test'] = [train_data['test'][i]for i in range(math.ceil(args.data_num * 15 / 100))]       
    
 
    if args.hyperparameter_search:
        print("\n ==> Start hyper-parameter search")

       

        def objective(trial):
            run_training(args=args, train_data=train_data, tokenizer=tokenizer, trial=trial)            
            # You could define your own compute_objective function, if not defined, the default compute_objective will be called,
            # and the sum of eval metric like f1 is returned as objective value.
            #return run_training(args=args, train_data=train_data, tokenizer=tokenizer, trial=trial)  
            # return evaluation_score ist richtig 
            
            #results = trainer.evaluate(eval_dataset=finaltest_set)     

            def compute_metrics(eval_pred):        
                print("\n DDDDcompute_metrics: ")
                predictions, labels = eval_pred 
                tuple_element_1 = np.asarray(predictions[0])
                tuple_element_2 = np.asarray(predictions[1])
                predictions = np.argmax(tuple_element_1, axis=1)
                clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])    
                return clf_metrics.compute(predictions=predictions, references=labels)     
            

            accuracy = evaluate.load("accuracy")
            f1_metric = evaluate.load("f1")
            precision = evaluate.load("precision")
            recall = evaluate.load("recall")
            print("f1: ", f1_metric)
            
            #results = f1_metric.compute(references=references, predictions=predictions, average=None)
            #print(results) # das ist das, wo die Beispiele rauskommen
            
            #f_test = compute_metrics(f1_metric)
            #print("f_test: ", f_test)
            #print("f_test 1", f_test[1])

            
       
            #trial.report(f1, epoch)

            # Handle pruning based on the intermediate value.
            #if trial.should_prune():
                #raise optuna.exceptions.TrialPruned()  

            # error because f1 is not defined
            #return f1
        
        study = optuna.create_study(
            #study_name='hp_search_' + args.vuln_type,
            direction="maximize",
            storage='sqlite:///hp_search.db',
            #load_if_exists=True
            )
        test = study.optimize(objective, n_trials=args.n_trials)
        print("test: ", test)
        print("Best trial:")
        trial = study.best_trial

        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    else:
        print("\n  ==> Started fine-tuning")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.load, 
            num_labels=2,
            id2label={0: "NOT VULNERABLE", 1: "VULNERABLE"},
            label2id={"NOT VULNERABLE": 0, "VULNERABLE": 1}
        )
        print("\n 1. loaded model")
        print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")
        run_training(args=args, train_data=train_data, tokenizer=tokenizer, model=model)


if __name__ == "__main__": 
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
    parser.add_argument('--grad-acc-steps', default=4, type=int) # instead of updating the model parameters after processing each batch, macht also normale batch size obsolet
    parser.add_argument('--local_rank', default=-1, type=int) # irgendwas mit distributed training
    parser.add_argument('--deepspeed', default=None, type=str) # intetration with deepspeed library
    parser.add_argument('--fp16', default=False, action='store_true') # with mixed precision for training acceleration

    # Logging and stuff
    parser.add_argument('--save-dir', default="../hp_search/", type=str)
    parser.add_argument('--log-freq', default=10, type=int)# commented out for gpu
    parser.add_argument('--save-freq', default=500, type=int)       # default = 500 # commented out for gpu

    # Hyperparameter search
    parser.add_argument('--hyperparameter-search', action='store_true')
    parser.add_argument('--n_trials', default=10, type=int)
   

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
