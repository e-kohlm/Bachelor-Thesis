from datasets import load_dataset
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding
                        )
import evaluate
import numpy as np




#https://huggingface.co/docs/transformers/tasks/sequence_classification



file_path = "../VUDENC_data/"
training_set = "EXAMPLE_sql_dataset-TRAINING"
validation_set = "EXAMPLE_sql_dataset-VALIDATION"
test_set = "EXAMPLE_sql_dataset-TESTING"
raw_datasets = "EXAMPLE_sql_dataset"
data_files = {"train": file_path + training_set, "validation": file_path + validation_set, "test": file_path + test_set}
datasets = load_dataset("json", data_files=data_files)
print("datasets: ", datasets)
print("datasets test: ", datasets['test'][0], "\n" )


checkpoint = "Salesforce/codet5p-220m"
device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def preprocess_function(examples):
    print("examples 0: ", examples["code"][0])
    print("examples 53: ", examples["code"][53])
    print("examples 2000: ", examples["code"][200])
    return tokenizer(examples["code"], truncation=True)

def test_preprocess_function(datasets):
    print("datasets 0: ", datasets["code"][0])
    print("datasets 53: ", datasets["code"][53])
    print("datasets 2010: ", datasets["code"][200])
    return tokenizer(datasets["code"], truncation=True)    



tokenized_datasets = datasets.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")
print("accuracy examples: ", accuracy, "\n")


tokenized_test_datasets = datasets.map(test_preprocess_function, batched=True)   
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")
print("accuracy datasets: ", accuracy)


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)