import jsonlines
import torch.nn as nn
from tqdm import tqdm
import pickle
import torch
from sentence_transformers import SentenceTransformer
import utils.dataset_generate as dg
import numpy as np


class Data_Process:
    def __init__(self):
        self.mqlen = 0

    def json2dataset(self, filepath, datasetpath):
        # json_list = []
        # with open(filepath, "r+") as f_in:
        #     for json in jsonlines.Reader(f_in):
        #         if len(json_list) < 40:
        #             json_list.append(json)
        # model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        # model = nn.DataParallel(model)
        # # model = model.to("cuda")
        # mq_raw = [item["main_question"] for item in json_list]
        # mq_embedding = model.module.encode(mq_raw, show_progress_bar=True)
        # mq_embedding = mq_embedding.tolist()
        # # with open("../scratch/data/mq", "rb") as fp:
        # # mq_embedding=pickle.load(fp)
        # x = mq_embedding
        # # with open("../scratch/data/mq", "wb") as fp:
        # #     pickle.dump(mq_embedding, fp)
        # ind_point = len(mq_embedding) - 1
        # print(ind_point)
        # y = [i for i in range(ind_point + 1)]

        # rq_hash = []
        # edge_list = [[], []]
        # for mq_ind, item in tqdm(enumerate(json_list)):
        #     for rq_item in item["related_questions"]:
        #         if rq_item in mq_raw:
        #             if mq_raw.index(rq_item) not in rq_hash:
        #                 rq_hash.append(mq_raw.index(rq_item))
        #             edge_list[0].append(mq_ind)
        #             edge_list[1].append(mq_raw.index(rq_item))
        #         else:
        #             ind_point += 1
        #             rq_hash.append(ind_point)
        #             edge_list[0].append(mq_ind)
        #             edge_list[1].append(ind_point)
        #             x.append(model.module.encode(rq_item))
        #             y.append(ind_point)

        # start_ind = ind_point
        # with open("../scratch/data/sq_cls06", "rb") as fp:
        #     sq_er = pickle.load(fp)
        # sq_hash = {}
        # for mq_ind, item in tqdm(enumerate(json_list)):
        #     for sq_item in item["sub_questions"][0]:
        #         current_loc = ind_point - start_ind
        #         current_cls = np.where(sq_er == sq_er[current_loc])
        #         if len(current_cls) > 1:
        #             if current_cls[0] != current_loc:
        #                 edge_list[0].append(mq_ind)
        #                 edge_list[1].append(sq_hash[current_cls[0]])
        #                 mq_ind = sq_hash[current_cls[0]]
        #                 x.append(model.module.encode(sq_item))
        #                 y.append(sq_hash[current_cls[0]])
        #             else:
        #                 ind_point += 1
        #                 sq_hash[current_loc] = ind_point
        #                 edge_list[0].append(mq_ind)
        #                 edge_list[1].append(ind_point)
        #                 mq_ind = ind_point
        #                 x.append(model.module.encode(sq_item))
        #                 y.append(ind_point)
        #         else:
        #             ind_point += 1
        #             edge_list[0].append(mq_ind)
        #             edge_list[1].append(ind_point)
        #             mq_ind = ind_point
        #             x.append(model.module.encode(sq_item))
        #             y.append(ind_point)

        # x = np.array(x)
        # edge_list = np.array(edge_list)
        # y = np.array(y)

        # with open("../scratch/data/x", "rb") as fp:
        #     x = pickle.load(fp)
        # fp.close()

        # with open("../scratch/data/ed", "rb") as fp:
        #     edge_list = pickle.load(fp)
        # fp.close()

        # with open("../scratch/data/y", "rb") as fp:
        #     y = pickle.load(fp)
        # fp.close()
        Dataset = dg.PKG_Dataset(
            datasetpath,
        )
        return Dataset
        # return