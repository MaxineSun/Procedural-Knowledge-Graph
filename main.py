import pickle
import utils.json2graph as jg
import utils.graph2vec as gv
import utils.parse_args as pa
import utils.sq_entity_resolution as sqer

from torchmetrics import RetrievalHitRate
from gensim.models import Word2Vec


def main(args):
    # process = jg.Data_Process()
    # issq = process.json2embedding(args.jsonpath)
    # train, val, test = process.embedding2graph()
    # nx_G = gv.read_graph(args)
    # G = gv.Graph(nx_G, args.directed, args.p, args.q)
    # G.preprocess_transition_probs()
    # walks = G.simulate_walks(args.num_walks, args.walk_length, process.mqlen)
    # gv.learn_embeddings(args, walks, train, val, test)
    process = sqer.Data_Process()
    sqlist = process.json2embedding(args.jsonpath)
    with open("../scratch/data/sqlist", "wb") as fp:
        pickle.dump(sqlist, fp)


if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
