from sentence_transformers import SentenceTransformer
import torch.nn as nn
import multiprocessing as mp
import pickle


class Data_Process:
    def __init__(self):
        self.sq_raw = []
        self.sq_ind = {}

    def json2sqlist(self, filepath):
        with open("../scratch/data/sqlist", "rb") as fp:
            self.sq_raw = pickle.load(fp)

        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model._modules['1'].word_embedding_dimension = 128
        model = nn.DataParallel(model)
        model = model.to('cuda')
        sq_embedding = model.module.encode(self.sq_raw[0])#, show_progress_bar=True)
        print(len(sq_embedding))
        with open("../scratch/data/sqemb128", "wb") as fp:
            pickle.dump(sq_embedding, fp)

        return
