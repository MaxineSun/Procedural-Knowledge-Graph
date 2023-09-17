from sentence_transformers import SentenceTransformer
import numpy as np
import torch.nn as nn
from sklearn.cluster import DBSCAN
import pickle


class DBScan_Process:
    def __init__(self):
        self.sq_raw = []
        self.sq_ind = {}

    def Clustering(self, density):
        with open("../scratch/data/json_list", "rb") as fp:
            json_list = pickle.load(fp)
        fp.close()
        for item in json_list:
            self.sq_raw += item["sub_questions"][0]
            
        with open("../scratch/data/sq_list", "wb") as fp:
            pickle.dump(self.sq_raw, fp)


        # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        # model._modules['1'].word_embedding_dimension = 128
        # model = nn.DataParallel(model)
        # model = model.to('cuda')
        # sq_embedding = model.module.encode(self.sq_raw, show_progress_bar=True)
        # sq_embedding = np.array(sq_embedding)
        
        # with open("../scratch/data/sq_embedding", "wb") as fp:
        #     pickle.dump(sq_embedding, fp)
    
        # sq_clustering = DBSCAN(eps=density, min_samples=1).fit(sq_embedding)
        # file_name = "sq_cls" + str(density)
        
    
        # with open("../scratch/data/" + file_name, "wb") as fp:
        #     pickle.dump(sq_clustering.labels_, fp)
        # return
 