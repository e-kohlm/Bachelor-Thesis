import os
import time
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
import json


def run_training(args, model, train_data):
    print(f"Starting main loop")
    start_time = time.time()

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')
    
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))


def load_tokenize_data(args):  # 5.   

    # Check if train_data already exists in cache_data/summarize_python
    if os.path.exists(args.cache_data):
        train_data = load_from_disk(args.cache_data)
        print(f'  ==> Loaded {len(train_data)} samples')
        return train_data
    # Load data
    else:        
        file_path = "../VUDENC_data/"
        training_set = "EXAMPLE_sql_dataset"        
        data_files = {file_path + training_set}
        datasets = load_dataset("json", data_files=data_files, split='train')
        print("datasets: ", datasets)       

        # Tokenize data
        tokenizer = AutoTokenizer.from_pretrained(args.load)

        def preprocess_function(examples):           
            
            return tokenizer(examples["code"], truncation=True)

        # Preprocess data
        train_data = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets.column_names,
            num_proc=64,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded {len(train_data)} samples')
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
    
    train_data = load_tokenize_data(args)  # 4. Daten werden geladen und tokenized
    
    print("args.data_num: ", args.data_num)
    if args.data_num != -1:        
        train_data = train_data.select([i for i in range(args.data_num)])

    
    # Load model from `args.load`
    #model = AutoModelForSequenceClassification.from_pretrained(args.load)
    #print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    #run_training(args, model, train_data)


if __name__ == "__main__":  # Argumente 
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on sequence classification task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=320, type=int)
    #parser.add_argument('--max-target-len', default=128, type=int)
    parser.add_argument('--cache-data', default='cache_data/summarize_python', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)  # TODO: auch mit codet5p-770m trainieren

    # Training
    parser.add_argument('--epochs', default=10, type=int)  # Number of times data set has to be worked through
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
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
