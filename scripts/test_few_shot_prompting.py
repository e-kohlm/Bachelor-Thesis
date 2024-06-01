import os
#import math
import json
#import pprint
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch


def load_create_data(args):      
    vulnerability_type = args.vuln_type  
      
    file_path = "../VUDENC_data/"
    data_set = file_path + vulnerability_type + "_dataset-TRAINING"
    with open(data_set, 'r') as file: 
        data = json.load(file)

    NOT_VULN_SAMPLES = 1  # todo statt hardgecoded als arg übergeben
    VULN_SAMPLES = 1 # todo statt hardgecoded als arg übergeben
    snippets_counter = 0
    vuln_snippets_counter = 0
    not_vuln_snippets_counter = 0
    samples =[] 
    for snippet in data:   
        snippets_counter += 1            
        snippet_id = snippet['snippet_id']
        print("snippet_id: ", snippet_id)    
        if snippet['label'] == 0:
            if not_vuln_snippets_counter < NOT_VULN_SAMPLES: 
                samples.append(snippet)    
                not_vuln_snippets_counter += 1
        elif snippet['label'] == 1:
            if vuln_snippets_counter < VULN_SAMPLES:
                samples.append(snippet)
                vuln_snippets_counter += 1 
    
    print("Number of snippets: ", snippets_counter)
    #print("samples: ", samples)
    print("type samples: ", type(samples))
    return samples
    
    
def format_prompt(samples, new_code, new_code_label):
    prompt = ""
    for sample in samples:
        print("sample: ", sample)
        prompt += f"{sample['code']}\nLabel: {sample['label']}\n"

    prompt += f"{new_code}\nLabel: {new_code_label}"
    return prompt


def main(args):
    argsdict = vars(args)  

    samples = load_create_data(args)
    #print("samples: ", samples)
    
    #new_code_snippet = "..."
    #prompt = format_prompt(samples, new_code_snippet)
    #print("prompt: ", prompt)
    torch.manual_seed(0)
    model = args.load
    tokenizer = AutoTokenizer.from_pretrained(model)
                                                         
    print(f"\n  ==> Loaded model from {args.load}")
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16)

    prompt = "Classify the code into 0 for not vulnerable or 1 for vulnerable. Code:INNER JOIN group_element_year_parent AS parent on parent.child_branch_id=child.parent_id ) SELECT * FROM group_element_year_parent ;  class GroupElementYearManager(models.Manager): def get_queryset' Label: " 
    sequences = pipe(prompt)
    for seq in sequences:
        print(f"Result:{seq['generated_text']}")

    """Output: 
    The model 'T5ForConditionalGeneration' is not supported for text-generation.

    Supported models are ['BartForCausalLM', 'BertLMHeadModel', 'BertGenerationDecoder', 'BigBirdForCausalLM', 'BigBirdPegasusForCausalLM',
    'BioGptForCausalLM', 'BlenderbotForCausalLM', 'BlenderbotSmallForCausalLM', 'BloomForCausalLM', 'CamembertForCausalLM', 'LlamaForCausalLM',
    'CodeGenForCausalLM', 'CpmAntForCausalLM', 'CTRLLMHeadModel', 'Data2VecTextForCausalLM', 'ElectraForCausalLM', 'ErnieForCausalLM',
    'FalconForCausalLM', 'FuyuForCausalLM', 'GemmaForCausalLM', 'GitForCausalLM', 'GPT2LMHeadModel', 'GPT2LMHeadModel', 'GPTBigCodeForCausalLM',
    'GPTNeoForCausalLM', 'GPTNeoXForCausalLM', 'GPTNeoXJapaneseForCausalLM', 'GPTJForCausalLM', 'LlamaForCausalLM', 'MarianForCausalLM',
    'MBartForCausalLM', 'MegaForCausalLM', 'MegatronBertForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM', 'MptForCausalLM', 'MusicgenForCausalLM',
    'MvpForCausalLM', 'OpenLlamaForCausalLM', 'OpenAIGPTLMHeadModel', 'OPTForCausalLM', 'PegasusForCausalLM', 'PersimmonForCausalLM', 'PhiForCausalLM',
    'PLBartForCausalLM', 'ProphetNetForCausalLM', 'QDQBertLMHeadModel', 'Qwen2ForCausalLM', 'ReformerModelWithLMHead', 'RemBertForCausalLM',
    'RobertaForCausalLM', 'RobertaPreLayerNormForCausalLM', 'RoCBertForCausalLM', 'RoFormerForCausalLM', 'RwkvForCausalLM', 'Speech2Text2ForCausalLM',
    'StableLmForCausalLM', 'TransfoXLLMHeadModel', 'TrOCRForCausalLM', 'WhisperForCausalLM', 'XGLMForCausalLM', 'XLMWithLMHeadModel', 'XLMProphetNetForCausalLM',
    'XLMRobertaForCausalLM', 'XLMRobertaXLForCausalLM', 'XLNetLMHeadModel', 'XmodForCausalLM'].
    /vol/fob-vol5/mi14/kohlmane/.local/lib/python3.11/site-packages/transformers/generation/utils.py:1178: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
    warnings.warn(
    Result:Classify the code into 0 for not vulnerable or 1 for vulnerable. Code:INNER JOIN group_element_year_parent AS parent on
    parent.child_branch_id=child.parent_id ) SELECT * FROM group_element_year_parent ;  class GroupElementYearManager(models.Manager):
    def get_queryset' Label: """


    









        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="CodeT5+ few-shot prompting on sequence classification task")
    parser.add_argument('--vuln-type', default="sql", type=str)  
    #parser.add_argument('--data-num', default=-1, type=int)  
    parser.add_argument('--cache-data', default='../cache_data', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str) 

    # Training
    #parser.add_argument('--epochs', default=10, type=int) # epochs
    #parser.add_argument('--lr', default=5e-5, type=float) # learning rate
    #parser.add_argument('--lr-warmup-steps', default=200, type=int) # learning rate
    #parser.add_argument('--batch-size-per-replica', default=8, type=int) # nicht dasselbe wie batch size, denke ich
    #parser.add_argument('--batch-size', default=256, type=int)  #   nicht aus ursprünglichem fine-tuning sondern andere Stelle codeT5 
    #parser.add_argument('--grad-acc-steps', default=4, type=int) # instead of updating the model parameters after processing each batch, macht also normale batch size obsolet
    #parser.add_argument('--local_rank', default=-1, type=int) # irgendwas mit distributed training
    #parser.add_argument('--deepspeed', default=None, type=str) # intetration with deepspeed library
    #parser.add_argument('--fp16', default=False, action='store_true') # with mixed precision for training acceleration

    # Logging and stuff
    parser.add_argument('--save-dir', default="../saved_few_shot_stuff/", type=str)
    #parser.add_argument('--log-freq', default=10, type=int)
    #parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
    