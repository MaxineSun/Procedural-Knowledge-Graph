import jsonlines
import torch
from tqdm import tqdm
import networkx as nx
from sentence_transformers import SentenceTransformer, util


class Data_Process:
    def __init__(self):
        self.mq = None
        self.rq = {}
        self.sq = {}

    def json2embedding(self, filepath):
        json_list = []
        with open(filepath, 'r+') as f_in:
            for json in jsonlines.Reader(f_in):
                json_list.append(json)

        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
        mq_raw = [item["main_question"] for item in json_list]
        mq_embedding = model.encode(mq_raw, show_progress_bar=True)
        self.mq = {ind: emb for ind, emb in enumerate(mq_embedding)}
        ind_point = len(mq_embedding)

        rq_raw = [item["related_questions"] for item in json_list]
        for ind_mq, rq_items in tqdm(enumerate(rq_raw)):
            tmp_rq = {}
            for rq_item in rq_items:
                item_embedding = model.encode(rq_item)
                if rq_item in mq_raw:
                    tmp_rq[mq_raw.index(rq_item)] = item_embedding
                else:
                    tmp_rq[ind_point] = item_embedding
                    ind_point += 1
            self.rq[ind_mq] = tmp_rq

        sq_raw = [item["sub_questions"][0] for item in json_list]
        for ind_mq, sq_items in tqdm(enumerate(sq_raw)):
            tmp_sq = {}
            for sq_item in sq_items:
                item_embedding = model.encode(sq_item)
                tmp_sq[ind_point] = item_embedding
                ind_point += 1
            self.sq[ind_mq] = tmp_sq

        return

    def embedding2graph(self):
        G = nx.Graph()
        for mq_ind in tqdm(range(len(self.mq))):
            for rq_ind, rq_emb in self.rq[mq_ind].items():
                if not G.has_edge(*(mq_ind, rq_ind)):
                    weight = float(util.cos_sim(self.mq[mq_ind], rq_emb))
                    G.add_edge(mq_ind, rq_ind, weight=weight)
            pre_sq_ind = mq_ind
            for sq_ind, sq_emb in self.sq[mq_ind].items():
                weight = float(util.cos_sim(self.mq[mq_ind], sq_emb))
                G.add_edge(pre_sq_ind, sq_ind, weight=weight)
                pre_sq_ind = sq_ind
        nx.write_weighted_edgelist(G, "data/graph/wikihow.edgelist")
