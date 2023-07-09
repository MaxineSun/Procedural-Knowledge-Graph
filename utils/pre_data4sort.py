import jsonlines
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from torch_geometric.data import Data


class Data_Process:
    def __init__(self):
        self.mqlen = 0

    def json2dataset(self, filepath):
        # json_list = []
        # with open(filepath, "r+") as f_in:
        #     for json in jsonlines.Reader(f_in):
        #         if len(json_list)<800:
        #             json_list.append(json)
        with open("../scratch/data/json_list", "rb") as fp:
            json_list = pickle.load(fp)
        fp.close()
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        model = nn.DataParallel(model)
        model = model.to("cuda")

        for mq_ind, item in tqdm(enumerate(json_list)):
            y = []
            edge_list = [[], []]
            ind_point = 0
            previous_ind = 0
            x = model.module.encode(item["main_question"])
            y.append(0)
            for sq_ind, sq_item in enumerate(item["sub_questions"][0]):
                if sq_ind < 10:
                    ind_point +=1
                    edge_list[0].append(previous_ind)
                    edge_list[1].append(ind_point)
                    previous_ind = ind_point
                    x = np.vstack((x, model.module.encode(sq_item)))
                    y.append(sq_ind+1)
            x = np.array(x)
            edge_list = np.array(edge_list)
            y = np.array(y)

            subgraph = Data(
                x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_list),
                y=torch.from_numpy(y),
            )

            with open(f"../scratch/data/subgraphs/g{str(mq_ind).zfill(5) }", "wb") as fp:
                pickle.dump(subgraph, fp)
            fp.close()

        return