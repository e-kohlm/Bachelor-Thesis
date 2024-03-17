from datasets import load_dataset
from transformers import (
                            AutoTokenizer,
                            DataCollatorWithPadding,                            
                            AutoModelForSequenceClassification,                            
                            TrainingArguments,
                            Trainer, 
                            pipeline
                        )
import evaluate
import torch
import numpy as np

from contextlib import redirect_stdout # weg


"""
Just for understanding how it works.
Don't change it!!!
It is the tutorial with a BERT model
"""

#https://huggingface.co/docs/transformers/tasks/sequence_classification
#Guide shows how to finetune DistilBERT and use your finetuned model for inference
#läuft so wie es ist durch, output gesichert in: test_outputs/sequence_classification_output.txt



file_path = "../VUDENC_data/"
training_set = "EXAMPLE_sql_dataset-TRAINING"
validation_set = "EXAMPLE_sql_dataset-VALIDATION"
test_set = "EXAMPLE_sql_dataset-TESTING"
raw_datasets = "EXAMPLE_sql_dataset"
data_files = {"train": file_path + training_set, "validation": file_path + validation_set, "test": file_path + test_set}
datasets = load_dataset("json", data_files=data_files)
#print("datasets: ", datasets)
print("datasets test: ", datasets['train'][0], "\n" )

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


#checkpoint = "distilbert/distilbert-base-uncased"
#device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage
#tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}
label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1}
#model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, id2label=id2label, label2id=label2id)
model = AutoModelForSequenceClassification.from_pretrained(
                                                            "distilbert/distilbert-base-uncased",
                                                            num_labels=2,
                                                            id2label=id2label,
                                                            label2id=label2id
                                                            )

print("tokenizer model_max_length: ", tokenizer.model_max_length)                                                            

def preprocess_function(examples):    
    with open('test_outputs/test_tutorial_sequence_classification_examples.txt', 'w') as f:
        with redirect_stdout(f):
            print("examples 0: ", examples["code"][0])
            print("examples 53: ", examples["code"][53])
            print("examples 200: ", examples["code"][200])    
    return tokenizer(examples["code"], truncation=True)  # truncate sequences to be no longer than DistilBERT’s maximum input length

# Only done if not saved already ... where is this written?
tokenized_datasets = datasets.map(preprocess_function, batched=True)  # You can speed up map by setting batched=True to process multiple elements of the dataset at once

# Now create a batch of examples using DataCollatorWithPadding. It’s more efficient to dynamically pad the sentences
# to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["snippet_id"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print("tokenized_datasets: ", tokenized_datasets)    

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


def compute_metrics(eval_pred):
    print("eval_pred: ", eval_pred)
    predictions, labels = eval_pred
    print("pred type: ", type(predictions))
    print("labels type: ", type(labels))
    print("predictions: ", predictions)
    print("labels: ", labels)
    print("pred shape: ", predictions.shape)
    print("labels shape: ", labels.shape)   
    predictions = np.argmax(predictions, axis=1)  # axis=1 nimmt das max der row    
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])    
    return clf_metrics.compute(predictions=predictions, references=labels)  # Saved in trainer_state.json of checkpooints
       

# Define your training hyperparameters
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


# Inference

#https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextClassificationPipeline
#https://huggingface.co/docs/transformers/task_summary#sequence-classification


# TODO evenn though, load_best_model_at_end=true, ohne checkpoint wird keine config.json gefunden
classifier_174 = pipeline(task="text-classification", model="my_awesome_model/checkpoint-174")
classifier_348 = pipeline(task="text-classification", model="my_awesome_model/checkpoint-348")
test_text = "tokenized_datasets = tokenized_datasets.remove_columns(['snippet_id'])\
            tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')\
            tokenized_datasets.set_format('torch')"
#print("test_text: ", test_text)
result_174 = classifier_174(test_text)
result_348 = classifier_348(test_text)
print("result_174: ", result_174)
print("result_348: ", result_348)
