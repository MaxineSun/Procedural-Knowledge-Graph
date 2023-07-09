import utils.parse_args as pa
# import utils.sq_dbscan as sqer
import utils.pre_data4sort as pd4s
# import utils.sq_clustering as sqer
import utils.json2heter as jh

from torchmetrics import RetrievalHitRate
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import pickle
# import utils.train 


def main(args):

    process = pd4s.Data_Process()
    process.json2dataset(args.jsonpath)


if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
