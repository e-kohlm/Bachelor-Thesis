import os
#os.environ["DS_SKIP_CUDA_CHECK"] = "1" #because of BUG: https://github.com/microsoft/DeepSpeed/issues/3223
#import time
from datetime import datetime, timedelta
import math
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import (AutoTokenizer,
                        TrainingArguments,
                        Trainer,                          
                        #pipeline,
                        #DataCollatorWithPadding,
                        AutoModelForSequenceClassification,
                        #logging
                        )
from load_tokenize_data import load_tokenize_data
from pynvml import *
from pynvml.smi import nvidia_smi
#import deepspeed
#from transformers.integrations import HfDeepSpeedConfig
#from accelerate import Accelerator
import json
import evaluate
import torch
import numpy as np
#import optuna
#from mpi4py import MPI
#os.environ['DS_SKIP_CUDA_CHECK']="1"

#logging.set_verbosity_error()

def print_gpu_utilization():
    nvmlInit()
    print("Driver Version:", nvmlSystemGetDriverVersion())
    deviceCount = nvmlDeviceGetCount()
    
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("Device", i, ":", nvmlDeviceGetName(handle))

    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"  GPU memory occupied: {info.used//1024**2} MB.")
    memory_reserved = torch.cuda.memory_reserved()
    print(f"GPU memory reserved: {memory_reserved / 1024**3:.2f} GB")

    nvsmi = nvidia_smi.getInstance()
    device_query = nvsmi.DeviceQuery('memory.free, memory.total')
    print(f" Device query:{device_query}")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()





def run_training(args, model, train_data, tokenizer):    
    

    print_gpu_utilization()
    #start_time = time.time() 
    start_time = datetime.now()
    print("start_time: ", start_time)

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
    
    # Load DeepSpeed configuration
    #ds_config_file = '../deepspeed_config.json'
    #hf_deepspeed_config = HfDeepSpeedConfig(ds_config_file)


    training_args = TrainingArguments(              
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        do_eval=True, #neu and actually not necessary since eval_strategy is set
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
        dataloader_drop_last=True, #CodeT5+ default=True
        #dataloader_num_workers=4, # Number of subprocesses to use for data loading, default=0, 0 means that teh data will be loaded in the main process.
        
        local_rank=args.local_rank, # args.local_rank ist default
        #deepspeed=ds_config_file,
        deepspeed=args.deepspeed, # args.deepspeed ist default
        fp16=args.fp16,             # args.fp16 ist default
    )

    trainer = Trainer(
        model=model,
        args=training_args,        
        train_dataset=train_data["train"],
        eval_dataset=train_data["validation"],        
        tokenizer=tokenizer,        
        compute_metrics=compute_metrics,       

    ) 

    allocated_memory = torch.cuda.memory_allocated()
    print(f"\n  Current GPU memory allocated: {allocated_memory / 1024**3:.2f} GB")  # Convert bytes to GB for readability
    memory_reserved = torch.cuda.memory_reserved()
    print(f"  GPU memory reserved: {memory_reserved / 1024**3:.2f} GB\n")

    #print_gpu_utilization()
   
    something = trainer.train()
    nvidaiprint_summary(something)    
    trainer.save_model() # added by me later: worked with cpu, sql mit 'data_num': 5000 samples
    
    # Evaluate the model on the test dataset   
    finaltest_set = train_data['test']
    
    evaluation = trainer.evaluate(eval_dataset=finaltest_set)  
    prediction = trainer.predict(test_dataset=finaltest_set)
    print("evaluation: ", evaluation)
    print("prediction: ", prediction)

    with open(os.path.join(args.save_dir, str(str(args.vuln_type) + "_" + str(args.data_num) + "_evaluation_results.json")), "w") as f:
        json.dump(evaluation, f, indent=4)

    with open(os.path.join(args.save_dir, str(str(args.vuln_type) + "_" + str(args.data_num) + "_prediction_results.json")), "w") as f:
        json.dump(prediction.metrics, f, indent=4)

    if args.local_rank in [0, -1]: 
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")         
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')        
    
    
    #end_time = time.time()
    #time_elapsed = end_time - start_time
    #print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)),"\n" )
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    #print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)),"\n" )
    #time_elapsed_formatted = str(datetime.timedelta(seconds=time_elapsed))
    #print("time_elapsed_formatted: ", time_elapsed_formatted, "\n" )

    # Get days, hours, and minutes from the timedelta object
    days = time_elapsed.days
    hours = time_elapsed.seconds // 3600
    minutes = (time_elapsed.seconds % 3600) // 60
    # Format the output
    formatted_time_elapsed = f"Days:{days:02} Hours:{hours:02} Minutes:{minutes:02}"
    print(f"Time elapsed: {formatted_time_elapsed}")

# copied to load_tokenize_data.py should be imported - not tested yet!!!!
"""def load_tokenize_data(args, tokenizer):      
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
                 
            #return tokenizer(examples["code"], truncation=True, max_length=tokenizer.model_max_length, padding='max_length')
            return tokenizer(examples["code"], truncation=True, padding=True) #padding: True = pad to the longest sequence in the batch
                                                                              #truncation: True = truncate to the maximum length accepted by the model if no max_length is provided                                                                                                                                              
        train_data = datasets.map(
            preprocess_function,
            #data_collator, #neu
            batched=True,            
            #num_proc=16,           
        )    

        train_data = train_data.remove_columns(["snippet_id"])
        train_data = train_data.rename_column("label", "labels")
        train_data.set_format("torch")
        #print("train_data: ", train_data)         

        print(f'\n  ==> Tokenized {len(train_data)} samples')        
        train_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')
        return train_data
"""

def main(args): 
    
    argsdict = vars(args) 
    print("Arguments:\n", pprint.pformat(argsdict))
    
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))
 
    #tokenizer_max_len = 512                                        # mit order ohne das hat keine Auswirkung auf speed
    #tokenizer_config = {'max_len': tokenizer_max_len}    
    tokenizer = AutoTokenizer.from_pretrained(args.load) #, **tokenizer_config)
    #print("tokenizer: ", tokenizer)

    train_data = load_tokenize_data(args, tokenizer=tokenizer)  
    print("train_data: ", train_data)

    print("Before model loaded: ")
    print_gpu_utilization()

    # Check if an argument to test a smaller sample of data was given
    if args.data_num != -1:             
        train_data['train'] = [train_data['train'][i]for i in range(math.ceil(args.data_num * 70 /100))]        
        train_data['validation'] = [train_data['validation'][i]for i in range(math.ceil(args.data_num * 15 / 100))]
        train_data['test'] = [train_data['test'][i]for i in range(math.ceil(args.data_num * 15 / 100))]       
    
 
    id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}   
    label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1} 
    #accelerator = Accelerator()
    #device = accelerator.device


    
    #device = torch.device("cuda:1") # wenn die 2 anderen in Betrieb, dann aber auch oben bei data_loader ändern




    device = "cuda" # "cuda" for GPU usage or "cpu" for CPU usage     
    model = AutoModelForSequenceClassification.from_pretrained(args.load, #full xss heute nur 147 und sql nur 149 Stunden???
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id).to(device)
    print("After model loaded: ")
    print_gpu_utilization()                                                        
    
                                                         
    print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data, tokenizer=tokenizer)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on sequence classification task")
    parser.add_argument('--vuln_type', default="sql", type=str)  
    parser.add_argument('--data_num', default=-1, type=int)  
    parser.add_argument('--cache_data', default='../cache_data', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str) 

    # Training    
    parser.add_argument('--lr', default=2e-5, type=float) # initial learning rate for AdamW HF: default=5e-5
    parser.add_argument('--lr_warmup_steps', default=0, type=int) # codet5+ paper: -, codet5+ code: default=200
    parser.add_argument('--per_device_train_batch_size', default=32, type=int) 
    parser.add_argument('--per_device_eval_batch_size', default=8, type=int)  # Added for OutOfMem issue, HF: default=8;  neu!!! Reduce when OutOfMem occurs; does not need to have same value as per_device_train_batch_size
    parser.add_argument('--optimizer', default='adamw_torch', type=str) # HF: default=adamw_torch
    parser.add_argument('--epochs', default=10, type=int) # epochs
    parser.add_argument('--weight_decay', default=0.1, type=int) # HF: default=0, codet5+ code: default=0.05
    parser.add_argument('--grad_acc_steps', default=1, type=int) # codet5+ code: default=4; instead of updating the model parameters after processing each batch, macht also normale batch size obsolet
    
    # Tokenization
    #parser.add_argument('--max_source_len', default=320, type=int) # codet5+ code: default=320
    #parser.add_argument('--max_target_len', default=128, type=int) # codet5+ code: default=128
   
    # GPU / Speeding up
    parser.add_argument('--local_rank', default=-1, type=int) # default=-1, irgendwas mit distributed training
    parser.add_argument('--deepspeed', default=None, type=str) # interacion with deepspeed library default = None, ("deepspeed_config.json",)
    parser.add_argument('--fp16', default=False, action='store_true') # default=False, action='store_true', with mixed precision for training acceleration

    # Logging and stuff
    parser.add_argument('--save_dir', default="../saved_models", type=str)
    parser.add_argument('--log_freq', default=10, type=int) #default=10
    parser.add_argument('--save_freq', default=500, type=int) # default=500 

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
    