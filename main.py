import utils.json2dataset as jd
import utils.parse_args as pa
# import utils.sq_dbscan as sqer
import utils.sq_ER_GPT as sqer
# import utils.sq_clustering as sqer

from torchmetrics import RetrievalHitRate
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import pickle
# import utils.train 


def main(args):
    # process = sqer.DBScan_Process()
    # process.Clustering(args.density)
    # process = sqer.GPT_Val_Process()
    # process.gpt_val()

    process = jd.Data_Process()
    process.json2dataset(args.jsonpath, args.datasetpath)
    # with open("../scratch/data/processed/dataset", "wb") as fp:
    #     pickle.dump(dataset, fp)
    # fp.close()

    # with open("../scratch/data/dataset", "rb") as fp:
    #     dataset = pickle.load(fp)
    # fp.close()


if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
