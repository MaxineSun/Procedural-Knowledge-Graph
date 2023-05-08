import jsonlines
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Data_Process:
    def __init__(self):
        self.sq_raw = []

    def json2embedding(self, filepath):
        json_list = []
        with open(filepath, 'r+') as f_in:
            for json in tqdm(jsonlines.Reader(f_in)):
                # if len(json_list) >= 30:
                #     break
                json_list.append(json)

        for items in tqdm(json_list):
            for item in items["sub_questions"][0]:
                self.sq_raw.append(item)

        # openai.api_key = "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"
        #
        # similarity_matrix = np.zeros((len(self.sq_raw), len(self.sq_raw)))
        # for i, text1 in enumerate(self.sq_raw):
        #     for j, text2 in enumerate(self.sq_raw):
        #         if i < j:
        #             similarity = cosine_similarity([openai.Embedding.create(input=[self.sq_raw[i]],
        #                                                                     model='text-embedding-ada-002')['data'][0][
        #                                                 'embedding']], [openai.Embedding.create(input=[self.sq_raw[j]],
        #                                                                                         model='text-embedding-ada-002')[
        #                                                                     'data'][0]['embedding']])[0][0]
        #             similarity_matrix[i, j] = similarity
        #             similarity_matrix[j, i] = similarity
        return self.sq_raw
