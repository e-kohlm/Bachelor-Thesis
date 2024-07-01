import os
import time
import math
import pprint
import argparse
from load_tokenize_data import load_tokenize_data
#from datasets import load_dataset, load_from_disk
from transformers import (AutoTokenizer,
                        TrainingArguments,
                        Trainer, 
                        AutoModelForSequenceClassification,
                        )
import json
import evaluate
import torch
import numpy as np


def run_training(args, model, train_data, tokenizer):    
    
    start_time = time.time() 

    #accuracy = evaluate.load("accuracy")
    #f1 = evaluate.load("f1")
    #precision = evaluate.load("precision")
    #recall = evaluate.load("recall")    


    def compute_metrics(eval_pred):        
        predictions, labels = eval_pred 
        tuple_element_1 = np.asarray(predictions[0])
        tuple_element_2 = np.asarray(predictions[1])
        predictions = np.argmax(tuple_element_1, axis=1)
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])    
        return clf_metrics.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(              
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        #do_eval=True, #neu and actually not necessary since eval_strategy is set
        do_predict=True,  #neu, whether to run predictions on the test set
        save_strategy='epoch', 
        eval_strategy="epoch",  
        metric_for_best_model="f1",       

        learning_rate=args.lr,
        warmup_steps=args.lr_warmup_steps,        
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        optim=args.optimizer,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_acc_steps,   

        save_total_limit=1,
        load_best_model_at_end=True, 
        save_only_model=True,
        logging_first_step=True,
        logging_steps=args.log_freq,
        logging_dir=args.save_dir,
        dataloader_drop_last=True,
        #dataloader_num_workers=4,  # Number of subprocesses to use for data loading, default=0, 0 means that teh data will be loaded in the main process.
        
        #use_cpu=True,  #macht vielleicht die Warning weg
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,  
    )

    trainer = Trainer(
        model=model,
        args=training_args,        
        train_dataset=train_data["train"],
        eval_dataset=train_data["validation"],          
        tokenizer=tokenizer,        
        compute_metrics=compute_metrics,       

    )   
   
    trainer.train()
    trainer.save_model() 

   

    # Evaluate the model on the test dataset   
    finaltest_set = train_data['test']

    #trainer.evaluate will predict + compute metrics on your test set
    #trainer.predict() will only predict labels on your test set. However in case the test set also contains ground-truth labels, the latter will also compute metrics.

    evaluation = trainer.evaluate(eval_dataset=finaltest_set)  
    prediction = trainer.predict(test_dataset=finaltest_set)
    print("evaluation: ", evaluation)
    print("prediction: ", prediction)

    with open(os.path.join(args.save_dir, str(str(args.vuln_type) + "_" + str(args.data_num) + "_evaluation_results.json")), "w") as f:
        json.dump(evaluation, f, indent=4)

    with open(os.path.join(args.save_dir, str(str(args.vuln_type) + "_" + str(args.data_num) + "_prediction_results.json")), "w") as f:
        json.dump(prediction.metrics, f, indent=4)

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint") #alt
        #final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint", "model.pt") # test: no
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')
    
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)),"\n" )

def main(args): 
    argsdict = vars(args) 
    print("Arguments:\n", pprint.pformat(argsdict))
   
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict)) 
        
    tokenizer = AutoTokenizer.from_pretrained(args.load) 

    train_data = load_tokenize_data(args, tokenizer=tokenizer)  

    # Check if an argument to test a smaller sample of data was given
    if args.data_num != -1:             
        train_data['train'] = [train_data['train'][i]for i in range(math.ceil(args.data_num * 70 /100))]        
        train_data['validation'] = [train_data['validation'][i]for i in range(math.ceil(args.data_num * 15 / 100))]
        train_data['test'] = [train_data['test'][i]for i in range(math.ceil(args.data_num * 15 / 100))]       
    
 
    id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}   
    label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1} 
    device = "cpu"  # "cuda" for GPU usage or "cpu" for CPU usage   
    model = AutoModelForSequenceClassification.from_pretrained(args.load, 
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id).to(device)
                                                              
    print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    # Verify model and tokenizer compatibility
    print(f"Model architecture: {model.config.architectures}")
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")

    run_training(args, model, train_data, tokenizer=tokenizer)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on sequence classification task")
    parser.add_argument('--vuln_type', default="sql", type=str)  
    parser.add_argument('--data_num', default=-1, type=int)  
    parser.add_argument('--cache_data', default='cache_data/', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str) 

    # Training hyperparameters (defaults=baseline model based on Wang-Le-Gotmare-etal 2023 CodeT5+)    
    parser.add_argument('--lr', default=2e-5, type=float) # initial learning rate for AdamW HF: default=5e-5
    parser.add_argument('--lr_warmup_steps', default=0, type=int) # codet5+ paper: -, codet5+ code: default=200
    parser.add_argument('--per_device_train_batch_size', default=32, type=int) # HF: default=8 
    parser.add_argument('--per_device_eval_batch_size', default=8, type=int)  # Added for OutOfMem issue, HF: default=8;  neu!!! Reduce when OutOfMem occurs; does not need to have same value as per_device_train_batch_size
    parser.add_argument('--optimizer', default='adamw_torch', type=str) # HF: default=adamw_torch
    parser.add_argument('--epochs', default=10, type=int) # codet5+ code: default=4, HF: default=3
    parser.add_argument('--weight_decay', default=0.1, type=int) # HF: default=0, codet5+ code: default=0.05
    parser.add_argument('--grad_acc_steps', default=1, type=int) # codet5+ code: default=4; instead of updating the model parameters after processing each batch, macht also normale batch size obsolet
        
    # Tokenization
    #parser.add_argument('--max_source_len', default=320, type=int) # codet5+ code: default=320
    #parser.add_argument('--max_target_len', default=128, type=int) # codet5+ code: default=128
    
    # GPU / Speeding up
    parser.add_argument('--local_rank', default=-1, type=int) # irgendwas mit distributed training
    parser.add_argument('--deepspeed', default=None, type=str) # interaction with deepspeed library, it is experimental
    parser.add_argument('--fp16', default=False, action='store_true') # with mixed precision for training acceleration

    # Logging
    parser.add_argument('--save_dir', default="saved_models/summarize_python", type=str)
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--save_freq', default=500, type=int)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
    