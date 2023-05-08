import jsonlines
import torch
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import openai


class Data_Process:
    def __init__(self):
        self.sq_raw = []

    def json2embedding(self, filepath):
        json_list = []
        with open(filepath, 'r+') as f_in:
            for json in tqdm(jsonlines.Reader(f_in)):
                # if len(json_list) >= 8:
                #     break
                json_list.append(json)

        for items in tqdm(json_list):
            for item in items["sub_questions"][0]:
                self.sq_raw.append(item)

        openai.api_key = "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"
        input_text = self.sq_raw[0]+' '+self.sq_raw[1]
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=input_text,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5,
        )

        embedding = response.choices[0].text.split("\n")[0]
        embedding = np.fromstring(embedding[1:-1], sep=", ")
        
        similarity = np.dot(embedding[:len(embedding)//2], embedding[len(embedding)//2:]) / (np.linalg.norm(embedding[:len(embedding)//2]) * np.linalg.norm(embedding[len(embedding)//2:]))

        print(similarity)