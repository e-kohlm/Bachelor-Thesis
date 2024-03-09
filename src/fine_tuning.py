import torch
from transformers import (
                        AutoTokenizer,
                        T5ForConditionalGeneration,
                        AutoModelForSeq2SeqLM,
                        AutoModelForSequenceClassification,
                        AdamW
                        )
from datasets import load_dataset
from datasets import load_dataset_builder
import numpy as np
import codecs
from contextlib import redirect_stdout
import time



file_path = "../VUDENC_data/"
training_set = 'EXAMPLE_sql_dataset-TRAINING'

# Load data and get some information about it
train_dataset = load_dataset("json", data_files=file_path + training_set, split='train')
print(train_dataset)
print(train_dataset.features)
print(train_dataset[0])
print(train_dataset[-1])

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

# CodeT5
checkpoint = "Salesforce/codet5p-220m"
device = "cpu" # "cuda" for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

# Code T5 from extra code on the right: TODO: welches soll ich nutzen???? wo finde ich solche Infos?
# Load model directly
#tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
#model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-220m")


# From Hugging Face working
"""inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# ==> print "Hello World"""

start_time = time.time()
i = 1
for snippet in train_dataset['code']:
    if i < 3:   # nur zum testen
        print("\n############################################")
        print("i: ", i)  # num_rows: 2770 == i
        print("snippet: ", snippet)

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



        inputs = tokenizer.encode(snippet, return_tensors="pt").to(device)
        print("inputs: ", inputs)       
        outputs = model.generate(model_inputs)

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
    
        i += 1


#print("outputs.last_hidden_state.shape: ", outputs.last_hidden_state.shape)
#AttributeError: 'Tensor' object has no attribute 'last_hidden_state'
#print("outputs.logits.shape: ", outputs.logits.shape)
##AttributeError: 'Tensor' object has no attribute 'logits'. Did you mean: 'logit'?
#print("outputs.logits: ", outputs.logits)
#predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#print("predictions: ", predictions)
#print("labels: ", model.config.id2label)


end_time = time.time()
elapsed_time = end_time - start_time
print("elapsed_time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#train a sequence classifier on one batch in PyTorch:




