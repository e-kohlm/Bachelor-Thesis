from datasets import load_dataset
from transformers import (
                            AutoTokenizer,
                            DataCollatorWithPadding,                            
                            AutoModelForSequenceClassification,                            
                            TrainingArguments,
                            Trainer
                        )
import evaluate
import torch
import numpy as np

#https://huggingface.co/docs/transformers/tasks/sequence_classification
#läuft so wie es ist durch, output gesichert in: test_outputs/sequence_classification_output.txt



file_path = "../VUDENC_data/"
training_set = "EXAMPLE_sql_dataset-TRAINING"
validation_set = "EXAMPLE_sql_dataset-VALIDATION"
test_set = "EXAMPLE_sql_dataset-TESTING"
raw_datasets = "EXAMPLE_sql_dataset"
data_files = {"train": file_path + training_set, "validation": file_path + validation_set, "test": file_path + test_set}
datasets = load_dataset("json", data_files=data_files)
#print("datasets: ", datasets)
print("datasets test: ", datasets['test'][0], "\n" )

# Find the max length of code snippet
"""i = 0
snippet_len_train =[] 
for snippet in datasets['train']:
    #print("i: ", i)
    #print("snippet['code']: ", snippet['code'])
    length = len(snippet['code'])
    snippet_len_train.append(length)
    #print("snippet_len_train: ", snippet_len_train)
    i += 1
print("i: ", i)
highest = max(snippet_len_train)
print("highest train: ", highest)  # highest train: 242

j = 0
snippet_len_validation =[] 
for snippet in datasets['validation']:
    #print("j: ", j)
    #print("snippet['code']: ", snippet['code'])
    length = len(snippet['code'])
    snippet_len_validation.append(length)
    #print("snippet_len_validation: ", snippet_len_validation)
    j += 1
print("j: ", j)
highest = max(snippet_len_validation)   
print("highest validation: ", highest)  # highest validation: 239

k = 0
snippet_len_test =[] 
for snippet in datasets['test']:
    #print("k: ", k)
    #print("snippet['code']: ", snippet['code'])
    length = len(snippet['code'])
    snippet_len_test.append(length)
    #print("snippet_len_test: ", snippet_len_test)
    k += 1
print("k: ", k)
highest = max(snippet_len_test)
print("highest test: ", highest)  # highest test: 243"""  
###################################

#checkpoint = "Salesforce/codet5p-220m"
checkpoint = "distilbert/distilbert-base-uncased"
device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}
label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1}
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, id2label=id2label, label2id=label2id)


def preprocess_function(examples):
    snippets_len = []
    length = len(examples['code'])
    snippets_len.append(length) 
    print("examples 0: ", examples["code"][0])
    print("examples 53: ", examples["code"][53])
    print("examples 200: ", examples["code"][200])    
    return tokenizer(examples["code"], truncation=True)

"""def test_preprocess_function(datasets):
    print("datasets 0: ", datasets["code"][0])
    print("datasets 53: ", datasets["code"][53])
    print("datasets 200: ", datasets["code"][200])
    return tokenizer(datasets["code"], truncation=True)  """  



tokenized_datasets = datasets.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")
print("accuracy examples: ", accuracy, "\n")

print ("xxxxxxxxxxxxxx")

"""tokenized_test_datasets = datasets.map(test_preprocess_function, batched=True)   
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")
print("accuracy datasets: ", accuracy)"""

tokenized_datasets = tokenized_datasets.remove_columns(["snippet_id"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

print("tokenized_datasets: ", tokenized_datasets)

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
