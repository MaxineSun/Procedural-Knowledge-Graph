import jsonlines
import torch.nn as nn
from tqdm import tqdm
import pickle
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader


class Data_Process:
    def __init__(self):
        self.mqlen = 0

    def json2dataset(self, filepath):
        # json_list = []
        # with open(filepath, "r+") as f_in:
        #     for json in jsonlines.Reader(f_in):
        #         if len(json_list) < 8000:
        #             json_list.append(json)
        # model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        # model = nn.DataParallel(model)
        # model = model.to("cuda")
        # mq_raw = [item["main_question"] for item in json_list]
        # mq_x = model.module.encode(mq_raw, show_progress_bar=True)
        # with open("../scratch/data/mq_x8", "wb") as fp:
        #     pickle.dump(mq_x, fp)
        # fp.close()

        # edge_mq_side = []
        # edge_sq_side = []
        # sq_cls_ind = 0
        # sq_node_ind = 0

        # with open("../scratch/data/sq_cls06", "rb") as fp:
        #     sq_cls = pickle.load(fp)
        # fp.close()

        # # """
        # # use entity resolution with dbscan density = 0.6
        # # """
        # sq_hash = {}
        # for mq_ind, item in tqdm(enumerate(json_list)):
        #     for sq_item in item["sub_questions"][0]:
        #         current_cls = np.where(sq_cls == sq_cls[sq_cls_ind])
        #         if sq_cls_ind == current_cls[0][0]:
        #             sq_emb = model.module.encode(sq_item)
        #             sq_hash[sq_cls_ind] = sq_node_ind
        #             sq_cls_ind += 1
        #             try:
        #                 sq_x
        #                 sq_x = np.vstack((sq_x, sq_emb))
        #             except NameError:
        #                 sq_x = sq_emb
        #             edge_mq_side.append(mq_ind)
        #             edge_sq_side.append(sq_node_ind)
        #             sq_node_ind += 1
        #         else:
        #             edge_mq_side.append(mq_ind)
        #             edge_sq_side.append(sq_hash[current_cls[0][0]])
        #             sq_cls_ind += 1
 
        # print(len(sq_x))
        # with open("../scratch/data/sq_x8", "wb") as fp:
        #     pickle.dump(sq_x, fp)
        # fp.close()

        # with open("../scratch/data/edge_mq_side", "wb") as fp:
        #     pickle.dump(edge_mq_side, fp)
        # fp.close()

        # with open("../scratch/data/edge_sq_side", "wb") as fp:
        #     pickle.dump(edge_sq_side, fp)
        # fp.close()

        with open("../scratch/data/mq_x", "rb") as fp:
            mq_x = pickle.load(fp)
        fp.close()

        print("mq data loaded")

        with open("../scratch/data/edge_mq_side", "rb") as fp:
            edge_mq_side = pickle.load(fp)
        fp.close()
        print("ed data loaded")


        with open("../scratch/data/edge_sq_side", "rb") as fp:
            edge_sq_side = pickle.load(fp)
        fp.close()
        print("sq ed data loaded")


        with open("../scratch/data/sq_x", "rb") as fp:
            sq_x = pickle.load(fp)
        fp.close()
        print("sq data loaded")


        data = HeteroData()
        data["mq"].node_id = torch.arange(len(mq_x))
        data["sq"].node_id = torch.arange(len(sq_x))
        data["mq"].x = torch.from_numpy(mq_x)
        data["sq"].x = torch.from_numpy(sq_x)
        data["mq","queries","sq"].edge_index = torch.stack([torch.tensor(edge_mq_side), torch.tensor(edge_sq_side)], dim=0)
        data = T.ToUndirected()(data)
        

        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=2.0,
            add_negative_train_samples=False,
            edge_types=("mq","queries","sq"),
            rev_edge_types=("sq","rev_queries","mq"),
        )
        train_data, val_data, test_data = transform(data)


        # Define seed edges:
        edge_label_index = train_data["mq","queries","sq"].edge_label_index
        edge_label = train_data["mq","queries","sq"].edge_label

        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=[20, 60],
            neg_sampling_ratio=2.0,
            edge_label_index=(("mq","queries","sq"), edge_label_index),
            edge_label=edge_label,
            batch_size=128,
            shuffle=True,
        )
        with open("../scratch/data/train_loaderI", "wb") as fp:
            pickle.dump(train_loader, fp)
        fp.close()

        edge_label_index = val_data["mq","queries","sq"].edge_label_index
        edge_label = val_data["mq","queries","sq"].edge_label
        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=[20, 60],
            edge_label_index=(("mq","queries","sq"), edge_label_index),
            edge_label=edge_label,
            batch_size=128,
            shuffle=False,
        )
        with open("../scratch/data/val_loaderI", "wb") as fp:
            pickle.dump(val_loader, fp)
        fp.close()
        print("val loader generated")

        edge_label_index = test_data["mq","queries","sq"].edge_label_index
        edge_label = test_data["mq","queries","sq"].edge_label
        test_loader = LinkNeighborLoader(
            data=test_data,
            num_neighbors=[20, 60],
            edge_label_index=(("mq","queries","sq"), edge_label_index),
            edge_label=edge_label,
            batch_size=128,
            shuffle=False,
        )

        with open("../scratch/data/test_loaderI", "wb") as fp:
            pickle.dump(test_loader, fp)
        fp.close()
        print("test loader generated")

        with open("../scratch/data/heter_dataI", "wb") as fp:
            pickle.dump(data, fp)
        fp.close()

        # return train_loader, val_loader, test_loader
        return
