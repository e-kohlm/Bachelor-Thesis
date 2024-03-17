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
Just for testing it is the tutorial of sequence classification copied and finetuned with CodeT5.
Not working, working version is fine_tuning.py
"""


#https://huggingface.co/docs/transformers/tasks/sequence_classification
#Guide shows how to finetune DistilBERT and use your finetuned model for inference
#läuft so wie es ist durch, output gesichert in: test_outputs/sequence_classification_output.txt



file_path = "../VUDENC_data/"
training_set = "EXAMPLE_sql_dataset-TRAINING"
validation_set = "EXAMPLE_sql_dataset-VALIDATION"
test_set = "EXAMPLE_sql_dataset-TESTING"
data_files = {"train": file_path + training_set, "validation": file_path + validation_set, "test": file_path + test_set}
datasets = load_dataset("json", data_files=data_files)    




checkpoint = "Salesforce/codet5p-220m"
device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}
label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1}
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, id2label=id2label, label2id=label2id)
#model = AutoModelForSequenceClassification.from_pretrained(
                                                            #"distilbert/distilbert-base-uncased",
                                                            #num_labels=2,
                                                            #id2label=id2label,
                                                            #label2id=label2id
                                                            #)

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
    """# ValueError: could not broadcast input array from shape (594,2) into shape (594,)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])    
    return clf_metrics.compute(predictions=predictions, references=labels)  # Saved in trainer_state.json of checkpooints"""
    predictions, labels = eval_pred        
    tuple_element_1 = np.asarray(predictions[0])
    tuple_element_2 = np.asarray(predictions[1])        
    print("tuple_element_1 shape: ", tuple_element_1.shape)
    print("tuple_element_2 shape: ", tuple_element_2.shape)
    print("labels shape: ", labels.shape)

    predictions = np.argmax(tuple_element_1, axis=1)  # FIXME only prediction with values of first tuple, what to do with the other one????
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])    
    return clf_metrics.compute(predictions=predictions, references=labels)  # TODO Save to file   

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
#classifier_cp_2 = pipeline(task="text-classification", model="saved_models/summarize_python")
classifier_f_cp = pipeline(task="text-classification", model="saved_models/summarize_python/final_checkpoint")
test_text = "tokenized_datasets = tokenized_datasets.remove_columns(['snippet_id']) tokenized_datasets = tokenized_datasets.rename_column('label', 'labels') tokenized_datasets.set_format('torch')"
vul_snippet = "SQL_RECURSIVE_QUERY_EDUCATION_GROUP='''\\ WITH RECURSIVE group_element_year_parent AS( SELECT id, child_branch_id, child_leaf_id, parent_id, 0 AS level FROM base_groupelementyear WHERE parent_id IN({list_root_ids'"
not_vul_snippet = "' INNER JOIN group_element_year_parent AS parent on parent.child_branch_id=child.parent_id ) SELECT * FROM group_element_year_parent ; ''''''''' class GroupElementYearManager(models.Manager): def get_queryset"
# TODO kein snippet reingeben, sondern viel Code, was passiert dann damit?
#print("test_text: ", test_text)
test_text_cp_2 = classifier_cp_2(test_text)
test_text_f_cp = classifier_f_cp(test_text)
vul_cp_2 = classifier_cp_2(vul_snippet)
vul_f_cp = classifier_f_cp(vul_snippet)
not_vul_snippet_cp_2 = classifier_cp_2(not_vul_snippet)
not_vul_snippet_f_cp = classifier_f_cp(not_vul_snippet)
print("test text cp 2: ", test_text_cp_2)
print("test texst f cp: ", test_text_f_cp)
print("vul cp 2: ", vul_cp_2)
print("vul f cp: ", vul_f_cp)
print("not vul cp 2: ", not_vul_snippet_f_cp)
print("not vul f cp: ", not_vul_snippet_f_cp)
