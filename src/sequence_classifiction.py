from datasets import load_dataset
from transformers import (
                            AutoTokenizer,
                            DataCollatorWithPadding,
                            T5ForConditionalGeneration,
                            AutoModelForSequenceClassification,
                            AutoModelForSeq2SeqLM,
                            TrainingArguments,
                            Trainer
                        )
import evaluate
import torch
import numpy as np




#https://huggingface.co/docs/transformers/tasks/sequence_classification



file_path = "../VUDENC_data/"
training_set = "EXAMPLE_sql_dataset-TRAINING"
validation_set = "EXAMPLE_sql_dataset-VALIDATION"
test_set = "EXAMPLE_sql_dataset-TESTING"
raw_datasets = "EXAMPLE_sql_dataset"
data_files = {"train": file_path + training_set, "validation": file_path + validation_set, "test": file_path + test_set}
datasets = load_dataset("json", data_files=data_files)
#print("datasets: ", datasets)
print("datasets test: ", datasets['test'][0], "\n" )


checkpoint = "Salesforce/codet5p-220m"
device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

id2label = {0: "NOT VULNERABLE", 1: "VULNERABLE"}
label2id = {"NOT VULNERABLE": 0, "VULNERABLE": 1}
#model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint, num_labels=2, id2label=id2label, label2id=label2id).to(device)

#model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True,
                                              trust_remote_code=True).to(device)  # was T5+ sagt???


def preprocess_function(examples):
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

print("tokenizec_datasets: ", tokenized_datasets)




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
