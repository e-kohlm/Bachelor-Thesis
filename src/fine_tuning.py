import torch
from transformers import (
                        AutoTokenizer,
                        T5ForConditionalGeneration,
                        AutoModelForSeq2SeqLM,
                        AutoModelForSequenceClassification,
                        AdamW,
                        TrainingArguments,
                        Trainer,
                        DataCollatorWithPadding
                        )
from datasets import load_dataset
from datasets import load_dataset_builder
import numpy as np
import evaluate
import codecs
from contextlib import redirect_stdout
from torch.utils.data import DataLoader
import time



file_path = "../VUDENC_data/"
#training_set = "EXAMPLE_sql_dataset-TRAINING"
#validation_set = "EXAMPLE_sql_dataset-VALIDATION"
raw_datasets = "EXAMPLE_sql_dataset"
"""
# Load data and get some information about it
train_dataset = load_dataset("json", data_files=file_path + training_set, split='train')
print(train_dataset)
print(train_dataset.features)
print(train_dataset[0])
print(train_dataset[-1])"""

# Working: from Hugging Face Tutorial
"""checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  # TODO: checkpoint = weights, which one?
tokenizer = AutoTokenizer.from_pretrained(checkpoint)   # TODO: it is not simply working with AutoAutoModelForQuestionAnswering
start_time = time.time()
i = 1
for snippet in train_dataset:
    print("\n############################################")
    print("i: ", i)  # num_rows: 2770 == i
    raw_inputs = train_dataset['code']
    '''with open('test_right_code.json', 'w') as f:    # überschreibt hier immer das ist okay
        with redirect_stdout(f):
            print("snippet['code']: ", snippet['code'], "\n*********\n")  
            print("snippet['label']: ", snippet['label'], "\n*********\n")'''
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") # pt means pytorch   
    #print("inputs: ", inputs, "\n############################################\n")
    i += 1
end_time = time.time()
elapsed_time = end_time - start_time
print("elapsed_time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))"""

"""# CodeT5
checkpoint = "Salesforce/codet5p-220m"
device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)"""

# Code T5 from extra code on the right: TODO: welches soll ich nutzen???? wo finde ich solche Infos?
# Load model directly
#tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
#model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-220m")


# From Hugging Face working
"""inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# ==> print "Hello World"""

"""i = 1
for snippet in train_dataset['code']:
    if i < 3:   # nur zum testen
        print("\n############################################")
        print("i: ", i)  # num_rows: 2770 == i
        print("snippet: ", snippet)"""

        # aus tutorial HF nlp/2/5: keine Ahnung, warum ich das brauche ...
        # ich habe ja unten schon auch tensors, mit return tensors ...
        # klappt auch nicht
        #tokens = tokenizer.tokenize(snippet)
        #print("tokens: ", tokens)
        #ids = tokenizer.convert_tokens_to_ids(tokens)
        #print("ids: ", ids)
        #input_ids = torch.tensor([ids])
        #print("input_ids: ", input_ids)
        #output = model(input_ids)
        #print("Logits: ", output.logits)
        ########################################

        #inputs = tokenizer.encode(snippet, return_tensors="pt").to(device)
        #print("inputs: ", inputs)       
        #outputs = model.generate(model_inputs)

        # UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length.
        # We recommend setting `max_new_tokens` to control the maximum length of the generation.

        # hf_outputs = model(**inputs)  # Nice try: hier wollte ich dieselbe Ausgabe machen wie im HF Turorial
        # ist aber trotzdem nicht unspannend, weil ich hier trotz Fehlermeldung Infos bekomme
        #print("outputs.last_hidden_state.shape: ", outputs.last_hidden_state.shape)
        #AttributeError: 'Tensor' object has no attribute 'last_hidden_state'
        #print("outputs.logit.shape: ", outputs.logits.shape)
        #AttributeError: 'builtin_function_or_method' object has no attribute 'shape'
        #print("outputs.logit: ", outputs.logits)
        #Output: outputs.logit:  <built-in method logits of Tensor object at 0x7f8929c6d730>
        #predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        #AttributeError: 'builtin_function_or_method' object has no attribute 'softmax'

        # i += 1

#print("outputs.last_hidden_state.shape: ", outputs.last_hidden_state.shape)
#AttributeError: 'Tensor' object has no attribute 'last_hidden_state'
#print("outputs.logits.shape: ", outputs.logits.shape)
##AttributeError: 'Tensor' object has no attribute 'logits'. Did you mean: 'logit'?
#print("outputs.logits: ", outputs.logits)
#predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#print("predictions: ", predictions)
#print("labels: ", model.config.id2label)

###########################################################


# Train a sequence classifier on one batch in PyTorch:
# Same as before
#checkpoint = "bert-base-uncased"
#tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# CodeT5

datasets = load_dataset(path="json", data_files=file_path + raw_datasets)
checkpoint = "Salesforce/codet5p-220m"
device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
#model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)


# train a sequence classifier
batch = tokenizer(datasets, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = torch.optim.AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()




def tokenize_function(datasets):
    print("tokenizer: ", tokenizer)
    return tokenizer(datasets['code'], truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True)
print("tokenized_datasets: ", tokenized_datasets)
tokenized_datasets = tokenized_datasets.remove_columns(["snippet_id"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
print("AGAIN tokenized_datasets: ", tokenized_datasets)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# test
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
print("samples: ", samples)


batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
print("batch: ", batch)

###########

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")




train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],  # findet der nicht, warum findet er train_dataset?
    data_collator=data_collator,
    tokenizer=tokenizer,
)

start_time = time.time()

trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)




end_time = time.time()
time_elapsed = end_time - start_time
print("time_elapsed: ", time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))        

        









sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]





