import jsonlines
import torch.nn as nn
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import utils.dataset_generate as DG
from torch_geometric.data import Dataset, Data


class Data_Process:
    def __init__(self):
        self.mqlen = 0

    def json2dataset(self, filepath):
        json_list = []
        # with open(filepath, "r+") as f_in:
        #     for json in jsonlines.Reader(f_in):
        #         if len(json_list)<8000:
        #             json_list.append(json)
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        model = nn.DataParallel(model)
        model = model.to("cuda")
        
        # mq_embedding = model.module.encode(mq_raw, show_progress_bar=True)
        with open("../scratch/data/json_list", "rb") as fp:
            json_list = pickle.load(fp)
        fp.close()
        with open("../scratch/data/mq_x", "rb") as fp:
            mq_embedding = pickle.load(fp)
        fp.close()
        
        mq_raw = [item["main_question"] for item in json_list]

        x = mq_embedding
        ind_point = len(mq_embedding) - 1
        y = [0]* len(x)

        rq_hash = []
        edge_list = [[], []]
        for mq_ind, item in tqdm(enumerate(json_list)):
            for rq_item in item["related_questions"]:
                if rq_item in mq_raw:
                    if mq_raw.index(rq_item) not in rq_hash:
                        rq_hash.append(mq_raw.index(rq_item))
                    edge_list[0].append(mq_ind)
                    edge_list[1].append(mq_raw.index(rq_item))
        
        for mq_ind, item in tqdm(enumerate(json_list)):
            for sq_ind, sq_item in enumerate(item["sub_questions"][0]):
                if sq_ind < 10:
                    ind_point +=1
                    edge_list[0].append(mq_ind)
                    edge_list[1].append(ind_point)
                    mq_ind = ind_point
                    x = np.vstack((x, model.module.encode(sq_item)))
                    y.append(sq_ind+1)

        x = np.array(x)
        edge_list = np.array(edge_list)
        y = np.array(y)

        with open("../scratch/data/x", "wb") as fp:
            pickle.dump(x, fp)
        fp.close()

        with open("../scratch/data/edge_list", "wb") as fp:
            pickle.dump(edge_list, fp)
        fp.close()
        
        with open("../scratch/data/y", "wb") as fp:
            pickle.dump(y, fp)
        fp.close()

        dataset = DG.PKG_Dataset("../scratch/data/graph_full")
        dataset.process()

        return