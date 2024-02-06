from torch.utils.data import DataLoader
from openicl.icl_retriever import BaseRetriever
from typing import List, Union, Optional
from openicl import PromptTemplate
from accelerate import Accelerator
import pickle
import pathlib
from transformers import AutoTokenizer
import torch
import re

def get_dataloader(datalist: List[List], batch_size: int) -> DataLoader:
    dataloader = DataLoader(datalist, batch_size=batch_size)
    return dataloader


def get_generation_prompt_list_from_retriever_indices(ice_idx_list: List[List[int]], retriever: BaseRetriever,
                                                      tokenizer, gen_field_replace_token: str,
                                                      max_model_token_num: Optional[int] = None,
                                                      ice_template: Optional[PromptTemplate] = None,
                                                      prompt_template: Optional[PromptTemplate] = None):
    prompt_list = []
    for idx, ice_idx in enumerate(ice_idx_list):
        ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
        prompt = retriever.generate_prompt_for_generate_task(idx, ice, gen_field_replace_token=gen_field_replace_token,
                                                             ice_template=ice_template, prompt_template=prompt_template)
        
        if max_model_token_num is not None and tokenizer is not None:
            prompt_token_num = get_input_token_num(tokenizer, prompt)
            while len(ice_idx) > 0 and prompt_token_num > max_model_token_num:
                ice_idx = ice_idx[:-1]
                ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
                prompt = retriever.generate_prompt_for_generate_task(idx, ice,
                                                                     gen_field_replace_token=gen_field_replace_token,
                                                                     ice_template=ice_template,
                                                                     prompt_template=prompt_template)
                prompt_token_num = get_input_token_num(tokenizer, prompt)
        prompt_list.append(prompt)
    return prompt_list


def get_input_token_num(tokenizer, input):
    return len(tokenizer(input, verbose=False)['input_ids'])


def get_arabic_number(prompt_list):
    pre_list = [item.split('####')[-1] for item in prompt_list]
    result_list = []
    for item in pre_list:
        arabic_number = re.findall(r'-?\d+\,?\.?\d*', item)
        if len(arabic_number)<1:
            arabic_number = ['']
        arabic_number = arabic_number[-1]
        result_list.append(arabic_number)
    return result_list


def get_ref_arabic_number(prompt_list):
    return [item.split('\n')[-1].replace("#### ","") for item in prompt_list]
    

def strict_acc(prediction, references):
    if len(prediction) != len(references):
        raise ValueError("The size of predictions is not correct!")
    count = 0
    for pre,ref in zip(prediction, references):
        if pre == ref:
            count += 1
    return count/len(prediction)

def soft_acc(prediction, references):
    if len(prediction) != len(references):
        raise ValueError("The size of predictions is not correct!")
    count = [0]*len(prediction)
    for i in range(len(prediction)):
        if prediction[i] in references[i] or references[i] in prediction[i]:
            count[i] = 1
    return sum(count)/len(prediction)
        
def shuffle_examples(entry):
    dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    model_dir = dir/"scratch"/"data"/"diff_sort"/"model_"
    with open(model_dir, "rb") as fp:
        model = pickle.load(fp)
    fp.close()
    prompt_list = entry[0].split("Question")[1:]
    prompt_list = ["Question"+item for item in prompt_list]
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    encoded_sqs = tokenizer(prompt_list[:-1], padding=True, truncation=True, max_length=128, return_tensors='pt')
    encoded_sqs = encoded_sqs.to('cuda')
    _, perm_matrix = model(encoded_sqs)
    idx_prompt = torch.tensor(list(range(len(encoded_sqs['input_ids']))), dtype=torch.float32).to('cuda')
    idx_prompt = torch.argsort(idx_prompt@perm_matrix).tolist()
    idx_prompt += [len(encoded_sqs['input_ids'])]
    sub_prompt_list = [prompt_join(prompt_list, idx_prompt)]
    return sub_prompt_list
    
def prompt_join(sep_prompt_list, idx):
    prompt_list = [sep_prompt_list[i] for i in idx]
    return '\n'.join(prompt_list)

def get_labbels(dataset, llm):
    if dataset == "dbpedia_14":
        # label_tokenids = ["company", "school", "artist", "player", "politics", "transport", "building", "nature", "village", "animal", "plant", "album", "film", "book"]
        if llm == "NousResearch/Llama-2-7b-hf":
            label_tokenids = [5001, 3762, 7664, 4847, 22661, 8608, 5214, 5469, 5720, 13019, 8024, 3769, 2706, 3143]
        if llm in ["gpt-j-6b", "gpt2-xl"]:
            label_tokenids = [39722, 1524, 6802, 2137, 4819, 4839, 2615, 3450, 7404, 5044, 4618, 5062, 2646, 1492]
    if dataset == "ag_news":
        # label_tokenids = ["word", "sports", "business", "science"]
        if llm == "NousResearch/Llama-2-7b-hf":
            label_tokenids = [1734, 14717, 5381, 10466]
        if llm in ["gpt-j-6b", "gpt2-xl"]:
            label_tokenids = [4775, 5701, 1597, 3783]
    if dataset == "CR":
        # label_tokenids = ["negative", "positive"]
        if llm == "NousResearch/Llama-2-7b-hf":
            label_tokenids = [8178, 6374]
        if llm in ["gpt-j-6b", "gpt2-xl"]:
            label_tokenids = [4633, 3967]
    if dataset == "sst2":
        # label_tokenids = ["negative", "positive"]
        if llm == "NousResearch/Llama-2-7b-hf":
            label_tokenids = [6374, 8178]
        if llm in ["gpt-j-6b", "gpt2-xl"]:
            label_tokenids = [3967, 4633]
    if dataset == "sst5":
        # label_tokenids = ["poor", "bad", "normal", "great", "perfect"]
        if llm == "NousResearch/Llama-2-7b-hf":
            label_tokenids = [16403, 4319, 20759, 1781, 2107]
        if llm in ["gpt-j-6b", "gpt2-xl"]:
            label_tokenids = [3595, 2089, 3487, 1049, 2818]
    return label_tokenids
    