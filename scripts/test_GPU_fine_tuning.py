import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" # not working
#os.environ['CUDA_VISIBLE_DEVICES']='2, 3' #notworking
import gc



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
import json
import evaluate
import torch
import numpy as np


# OutOfMemory issue auch mit batch-size-per-replica=1
# https://saturncloud.io/blog/how-to-solve-cuda-out-of-memory-error-in-pytorch/ sagt Hauptgrund ist ein zu großes model, was ich nicht lösen kann
# die gruenaus sind CUDA Y 11.8
# torch.cuda.OutOfMemoryError: CUDA out of memory.
# Tried to allocate 20.00 MiB. GPU 2 has a total capacity of 79.32 GiB of which 4.50 MiB is free.
# Process 32777 has 76.96 GiB memory in use. => das ist ein Prozess von wem anders auf gruenau!
# Including non-PyTorch memory, this process has 2.33 GiB memory in use. 
# Of the allocated memory 1.62 GiB is allocated by PyTorch, and 21.76 MiB is reserved by PyTorch but unallocated.

# nur mit ml env
#python -c "import torch; print(torch.cuda.get_device_name(0));"
#Tesla V100-PCIE-32GB


def run_training(args, model, train_data, tokenizer):    
    
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


    training_args = TrainingArguments(
      
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',
        #save_strategy="no", #neu
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
        push_to_hub=False, 
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
   
    trainer.train()
    trainer.save_model() # added by me later: worked with cpu, sql mit 'data_num': 5000 samples
    

    # https://discuss.huggingface.co/t/save-only-best-model-in-trainer/8442
    # You can set save_strategy to NO to avoid saving anything and save the final model once training is done with trainer.save_model()

    # Evaluate the model on the test dataset   
    finaltest_set = train_data['test']

    results = trainer.evaluate(eval_dataset=finaltest_set)  
    prediction = trainer.predict(test_dataset=finaltest_set)
    print("results: ", results)
    print("prediction: ", prediction)

    if args.local_rank in [0, -1]: # ggfs. ist final_checkpoint egal;
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint") #alt
        #final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint", "model.pt") # test: no
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')
        # todo: config.json pytorch_model.bin these two files are the model, see https://huggingface.co/learn/nlp-course/chapter2/3
    
    
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
        print("train_data: ", train_data)   

        
       


        

        print(f'\n  ==> Tokenized {len(train_data)} samples')        
        train_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')
        return train_data


def main(args): 
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
    #Finetune CodeT5+ models on any Seq2Seq LM tasks sagt CodeT5  
    """model = AutoModelForSeq2SeqLM.from_pretrained(args.load,   
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id)  # ValueError: not enough values to unpack (expected 2, got 1)"""
    """model = AutoModelForSequenceClassification.from_pretrained(args.load, #3170 hours   nach Mail: 911 hours
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id)"""
    

    #device = "cuda"  # for GPU usage or "cpu" for CPU usage
    test = torch.cuda.is_available()
    print("test: ", test) #output: true
    X_train = torch.FloatTensor([0., 1., 2.])
    test_zwei = X_train.is_cuda
    print("tess 2: ", test_zwei) # output: false
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)


    #device = torch.device('cuda:0')
    

    # todo: is T5ForSequ the right one?
    model = T5ForSequenceClassification.from_pretrained(args.load,   #slow as hell: 1665 hours, nach Mail: 908 hours 52.51s/it bei sql, 175 hours bei xss
                                                        num_labels=2,
                                                        id2label=id2label,
                                                        label2id=label2id).to(device)                                                         
    print(f"\n  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    #gc.collect() # not working
    # torch.cuda.empty_cache() not working
    train_data = torch.FloatTensor([0., 1., 2.])
    test_drei = train_data.is_cuda
    print("test_drei: ", test_drei) # output: false
    run_training(args, model, train_data, tokenizer=tokenizer)

    run: nice -n 19 python GPU_fine_tuning.py --vuln-type=sql --cache-data=cache_data/sql --batch-size-per-replica=4 --save-dir=saved_models/summarize_python/GPU/sql
    https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
    Error: train_dataset=train_data["train"],
                  ~~~~~~~~~~^^^^^^^^^
    IndexError: too many indices for tensor of dimension 1


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on sequence classification task")
    parser.add_argument('--vuln-type', default="sql", type=str)  
    parser.add_argument('--data-num', default=-1, type=int)  
    parser.add_argument('--cache-data', default='cache_data/', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str) 

    # Training
    parser.add_argument('--epochs', default=10, type=int) # epochs
    parser.add_argument('--lr', default=5e-5, type=float) # learning rate
    parser.add_argument('--lr-warmup-steps', default=200, type=int) # learning rate
    parser.add_argument('--batch-size-per-replica', default=8, type=int) # nicht dasselbe wie batch size, denke ich
    #parser.add_argument('--batch-size', default=256, type=int)  #   nicht aus ursprünglichem fine-tuning sondern andere Stelle codeT5 
    parser.add_argument('--grad-acc-steps', default=4, type=int) # instead of updating the model parameters after processing each batch, macht also normale batch size obsolet
    parser.add_argument('--local_rank', default=-1, type=int) # irgendwas mit distributed training
    parser.add_argument('--deepspeed', default=None, type=str) # intetration with deepspeed library
    parser.add_argument('--fp16', default=False, action='store_true') # with mixed precision for training acceleration

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/summarize_python", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
    