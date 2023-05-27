import numpy as np
from sklearn.cluster import DBSCAN
import pickle


class DBScan_Process:
    def __init__(self):
        self.sq_raw = []
        self.sq_ind = {}

    def Clustering(self, density):
        with open("../scratch/data/sqemb", "rb") as fp:
            sq_embedding = pickle.load(fp)
        sq_embedding = np.array(sq_embedding)
        sq_clustering = DBSCAN(eps=density, min_samples=1).fit(sq_embedding)
        file_name = "sq_cls" + str(density)
        with open("../scratch/data/" + file_name, "wb") as fp:
            pickle.dump(sq_clustering.labels_, fp)
        return
 