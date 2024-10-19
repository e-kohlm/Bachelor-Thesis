from transformers import (AutoTokenizer,
                          Trainer,
                          TrainingArguments,
                          AutoModelForSequenceClassification,
                          EarlyStoppingCallback)
import os
from datetime import datetime, timedelta
import time
import argparse
import math
import pprint
from load_tokenize_data import load_tokenize_data
from gpu_test import print_gpu_utilization
import numpy as np
import evaluate
import json


def run_search(args, train_data, tokenizer):
    start_time = time.time()

    device = args.device
    if device == "cuda":
        print("Before model loaded: ")
        print_gpu_utilization()

    def model_init(trial):
        id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}
        label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1}
        # todo: model is hardcoded, because model_init allows max. one argument 
        model_name = 'Salesforce/codet5p-220m'
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   num_labels=2,
                                                                   id2label=id2label,
                                                                   label2id=label2id).to(device)
        print(f"\n  ==> Loaded model from {model_name}, model size {model.num_parameters()}")
        return model

    def compute_objective(metrics):
        eval_result = trainer.evaluate()
        print("\n objective eval_result['eval_f1']", eval_result['eval_f1'])
        return eval_result['eval_f1']

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        tuple_element_1 = np.asarray(predictions[0])        
        tuple_element_2 = np.asarray(predictions[1])
        predictions = np.argmax(tuple_element_1, axis=1)
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        return clf_metrics.compute(predictions=predictions, references=labels)

    # Initial Search Space
    """def optuna_hp_space(trial):
        print(f'\n  ==> Started trial {trial.number}')
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "warmup_steps": trial.suggest_int('warmup_steps', 0, 500),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size",
                                                                     [8, 16, 32, 64, 128, 256, 1024]),                    
            "optim": trial.suggest_categorical("optim", ["adamw_torch", "rmsprop"]),
            "num_train_epochs": trial.suggest_int('epochs', 1, 50),
            "weight_decay": trial.suggest_float("weight_decay", 0.05, 0.1, log=True),
            "gradient_accumulation_steps": trial.suggest_int('gradient_accumulation_steps', 1, 8),
        } """   

    # Narrowed Search Space
    def optuna_hp_space(trial):
        print(f'\n  ==> Started trial {trial.number}')
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "warmup_steps": trial.suggest_int('warmup_steps', 0, 500),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64]),                    
            "optim": trial.suggest_categorical("optim", ["adamw_torch"]),
            "num_train_epochs": trial.suggest_int('epochs', 10, 20),
            "weight_decay": trial.suggest_float("weight_decay", 0.05, 0.1, log=True),
            "gradient_accumulation_steps": trial.suggest_int('gradient_accumulation_steps', 1, 8),
        }     

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        do_eval=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        metric_for_best_model="f1",         

        save_total_limit=1,
        load_best_model_at_end=True,
        save_only_model=False,
        logging_first_step=True,
        logging_steps=args.log_freq,
        logging_dir=args.save_dir,
        dataloader_drop_last=True,
        auto_find_batch_size=True,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,  
    )

    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=train_data["train"],
        eval_dataset=train_data["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        model_init=model_init,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        # early_stopping_threshold=0.0,
    )
    database_storage = os.path.join(args.save_dir, 'hp_search.db')
    best_run = trainer.hyperparameter_search(
        direction=["maximize"],
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=args.n_trials,
        storage=f'sqlite:///{database_storage}',
        compute_objective=compute_objective,
    )

    print("\n", best_run)
    with open(os.path.join(args.save_dir, str(str(args.vuln_type) + "_" + str(args.data_num) + "_best_run.json")), "w") as f:
        json.dump(best_run.metrics, f, indent=4)
    
    end_time = datetime.now()    
    time_elapsed = end_time - start_time  
    days = time_elapsed.days
    hours = time_elapsed.seconds // 3600
    minutes = (time_elapsed.seconds % 3600) // 60
    formatted_time_elapsed = f"Days:{days:02} Hours:{hours:02} Minutes:{minutes:02}"
    print(f"Time elapsed: {formatted_time_elapsed}")


def main(args):
    argsdict = vars(args)
    print("Arguments:\n", pprint.pformat(argsdict))

    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    tokenizer = AutoTokenizer.from_pretrained(args.load)
    train_data = load_tokenize_data(args=args, tokenizer=tokenizer)
    print("train_data:\n", train_data)    

    # Check if an argument to test a smaller sample of data was given
    if args.data_num != -1:
        train_data['train'] = [train_data['train'][i] for i in range(math.ceil(args.data_num * 70 / 100))]
        train_data['validation'] = [train_data['validation'][i] for i in range(math.ceil(args.data_num * 15 / 100))]
        train_data['test'] = [train_data['test'][i] for i in range(math.ceil(args.data_num * 15 / 100))]

    run_search(args=args, train_data=train_data, tokenizer=tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ hyperparameter search on sequence classification task")
    parser.add_argument('--vuln_type', default="EXAMPLE_sql", type=str)
    parser.add_argument('--data_num', default=-1, type=int)
    parser.add_argument('--cache_data', default='cache_data/', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)
    parser.add_argument('--device', default="cuda", type=str)  # "cuda" for GPU usage or "cpu" for CPU usage

    # Hyperparameter search
    parser.add_argument('--n_trials', default=1, type=int)    

    # GPU / Speeding up
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging
    parser.add_argument('--save_dir', default="../hyperparameter_search/", type=str)
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--save_freq', default=500, type=int)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
