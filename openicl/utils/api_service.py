import json
import requests
import os
import openai
from openai import OpenAI
import time
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
import transformers
import torch
import re

OPENICL_API_NAME_LIST = ['opt-175b', 'gpt3', 'llama', 'llama70b', 'gpt-3.5-turbo-instruct']
OPENICL_API_PARAMETER_DICT = {
    'opt-175b': ['URL', 'headers'],
    'gpt3': ['engine', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'sleep_time'],
    'llama': [],
    'llama70b': [],
    'gpt-3.5-turbo-instruct':[]
}
OPENICL_API_REQUEST_CONFIG = {
    'opt-175b': {
        'URL': "",  # http://xxx/completions or http://xxx/generate
        'headers': {
            "Content-Type": "application/json; charset=UTF-8"
        }
    },
    'gpt3': {
        'engine': "text-davinci-003",
        'temperature': 0,
        'max_tokens': 256,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'sleep_time': 3
    },
    'llama': {},
    'llama70b': {},
    'gpt-3.5-turbo-instruct':{}
}
PROXIES = {"https": "", "http": ""}


def is_api_available(api_name):
    if api_name is None:
        return False
    return True if api_name in OPENICL_API_NAME_LIST else False


def update_openicl_api_request_config(api_name, **kwargs):
    if api_name is None or not is_api_available(api_name):
        return

    parameter_list = OPENICL_API_PARAMETER_DICT[api_name]
    for parameter in parameter_list:
        if parameter in kwargs.keys():
            OPENICL_API_REQUEST_CONFIG[api_name][parameter] = kwargs[parameter]


def api_get_ppl(api_name, input_texts):
    if api_name == 'opt-175b':
        pyload = {"prompt": input_texts, "max_tokens": 0, "echo": True}
        response = json.loads(
            requests.post(OPENICL_API_REQUEST_CONFIG[api_name]['URL'], data=json.dumps(pyload),
                          headers=OPENICL_API_REQUEST_CONFIG[api_name]['headers'], proxies=PROXIES).text)
        lens = np.array([len(r['logprobs']['tokens']) for r in response['choices']])
        ce_loss = np.array([-sum(r['logprobs']['token_logprobs']) for r in response['choices']])
        return ce_loss / lens

    if api_name == 'gpt3':
        raise NotImplementedError("GPT-3 API doesn't support PPL calculation")


def api_get_tokens(api_name, input_texts):
    length_list = [len(text) for text in input_texts]

    if api_name == 'opt-175b':
        pyload = {"prompt": input_texts, "max_tokens": 100, "echo": True}
        response = json.loads(
            requests.post(OPENICL_API_REQUEST_CONFIG[api_name]['URL'], data=json.dumps(pyload),
                          headers=OPENICL_API_REQUEST_CONFIG[api_name]['headers'], proxies=PROXIES).text)
        return [r['text'] for r in response['choices']], [r['text'][length:] for r, length in
                                                          zip(response['choices'], length_list)]

    if api_name == 'gpt3':
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            engine=OPENICL_API_REQUEST_CONFIG['gpt3']['engine'],
            prompt=input_texts,
            temperature=OPENICL_API_REQUEST_CONFIG['gpt3']['temperature'],
            max_tokens=OPENICL_API_REQUEST_CONFIG['gpt3']['max_tokens'],
            top_p=OPENICL_API_REQUEST_CONFIG['gpt3']['top_p'],
            frequency_penalty=OPENICL_API_REQUEST_CONFIG['gpt3']['frequency_penalty'],
            presence_penalty=OPENICL_API_REQUEST_CONFIG['gpt3']['presence_penalty']
        )
        time.sleep(OPENICL_API_REQUEST_CONFIG['gpt3']['sleep_time'])
        return [(input + r['text']) for r, input in zip(response['choices'], input_texts)], [r['text'] for r in
                                                                                             response['choices']]
        
    if api_name == 'llama':
        model = 'NousResearch/Llama-2-7b-hf' #"meta-llama/Llama-2-7b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        sequences = pipeline(
            input_texts[0],
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1500,
        )
        result = sequences[0]['generated_text']
        result = result.split("\n Question")
        if len(result) < 5:
            result = sequences[0]['generated_text']
            result = result.split("\nQuestion")
        result = ["Question"+item for item in result]
        result[0] = result[0][9:]
        result = [item+"\n " for item in result]
        complete_output = [''.join(result[:5])]
        generated = result[4].split("Answer: Let's think step by step.")
        if len(generated)<2:
            generated = result[4].split("Answer:")
        generated = generated[1]
        generated = generated[:-2]
        return complete_output, [generated]
    
    
    if api_name == 'llama70b':
        model = 'NousResearch/Llama-2-7b-hf' #"meta-llama/Llama-2-7b-chat-hf"

        # tokenizer = AutoTokenizer.from_pretrained(model)
        
        model_path = "/cluster/work/sachan/shridhar/models/llama-70B"
        print("====")
        config = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        print("====")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        print("====")
        model = load_checkpoint_and_dispatch(model, model_path,
                                            device_map='auto',
                                            offload_folder="offload",
                                            offload_state_dict=True,
                                            dtype = "float16",
                                            no_split_module_classes=["LlamaDecoderLayer"])
        print("====")
        tokenizer = AutoTokenizer.from_pretrained(model)
        
        print(a)
        # AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(args.device)

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        sequences = pipeline(
            input_texts[0],
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=1500,
        )
        result = sequences[0]['generated_text']
        result = result.split("\n Question")
        if len(result) < 5:
            result = sequences[0]['generated_text']
            result = result.split("\nQuestion")
        result = ["Question"+item for item in result]
        result[0] = result[0][9:]
        result = [item+"\n " for item in result]
        complete_output = [''.join(result[:5])]
        generated = result[4].split("Answer: Let's think step by step.")
        if len(generated)<2:
            generated = result[4].split("Answer:")
        generated = generated[1]
        generated = generated[:-2]
        print(complete_output)
        print([generated])
        print(a)
        return complete_output, [generated]
    
        
    if api_name == 'gpt-3.5-turbo-instruct':
        client = OpenAI()
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=input_texts[0],
            max_tokens=1000
        )
        solution = '\n'+response.choices[0].text
        complete_output = [input_texts[0]+solution]
        return complete_output, [solution]