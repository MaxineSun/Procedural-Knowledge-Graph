import jsonlines
import torch.nn as nn
from tqdm import tqdm
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch.utils.data as data_utils


class Data_Process:
    def __init__(self):
        self.mq = None
        self.rq = {}
        self.sq = {}
        self.mqlen = 0

    def json2embedding(self, filepath):
        json_list = []
        with open(filepath, 'r+') as f_in:
            for json in jsonlines.Reader(f_in):
                if len(json_list) < 80:
                    json_list.append(json)
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model = nn.DataParallel(model)
        model = model.to('cuda')
        mq_raw = [item["main_question"] for item in json_list]
        mq_embedding = model.module.encode(mq_raw, show_progress_bar=True)
        mq_embedding = mq_embedding.tolist()
        self.mq = {ind: emb for ind, emb in enumerate(mq_embedding)}
        ind_point = len(mq_embedding)

        rq_raw = [item["related_questions"] for item in tqdm(json_list)]
        rq_embs = [model.module.encode(rq_item) for rq_item in tqdm(rq_raw)]
        for ind_mq, rq_items in tqdm(enumerate(rq_raw)):
            tmp_rq = {}
            for ind_rq, rq_item in enumerate(rq_items):
                if rq_item in mq_raw:
                    tmp_rq[mq_raw.index(rq_item)] = rq_embs[ind_mq][ind_rq]
                else:
                    tmp_rq[ind_point] = rq_embs[ind_mq][ind_rq]
                    ind_point += 1
            self.rq[ind_mq] = tmp_rq
        self.mqlen = ind_point

        issq = {}
        sq_raw = [item["sub_questions"][0] for item in tqdm(json_list)]
        sq_embs = [model.module.encode(sq_item) for sq_item in tqdm(sq_raw)]
        for ind_mq, sq_items in tqdm(enumerate(sq_raw)):
            tmp_sq = {}
            tmp_sq_ind = []
            for ind_sq in range(len(sq_items)):
                tmp_sq[ind_point] = sq_embs[ind_mq][ind_sq]
                tmp_sq_ind.append(ind_point)
                ind_point += 1
            self.sq[ind_mq] = tmp_sq
            issq[ind_mq] = tmp_sq_ind

        return issq

    def embedding2graph(self):
        mq_len = len(self.mq)
        dataset = range(mq_len)
        train_size = int(0.7*mq_len)
        val_size = int(0.2*mq_len)
        test_size = mq_len-train_size-val_size
        train_dataset, val_dataset, test_dataset = data_utils.random_split(
            dataset, [train_size, val_size, test_size])
        G = nx.Graph()
        for mq_ind in tqdm(train_dataset):
            for rq_ind, rq_emb in self.rq[mq_ind].items():
                if not G.has_edge(*(mq_ind, rq_ind)):
                    weight = float(util.cos_sim(self.mq[mq_ind], rq_emb))
                    G.add_edge(mq_ind, rq_ind, weight=weight)
            pre_sq_ind = mq_ind
            for sq_ind, sq_emb in self.sq[mq_ind].items():
                weight = float(util.cos_sim(self.mq[mq_ind], sq_emb))
                G.add_edge(pre_sq_ind, sq_ind, weight=weight)
                pre_sq_ind = sq_ind
        nx.write_weighted_edgelist(G, "../scratch/data/graph/wikihow.edgelist")
        return train_dataset, val_dataset, test_dataset
