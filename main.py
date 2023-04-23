import pickle
import utils.json2graph as jg
import utils.graph2vec as gv
import utils.parse_args as pa


def main(args):
    process = jg.Data_Process()
    process.json2embedding(args.jsonpath)

    # with open("data/mq", "wb") as fp:
    #     pickle.dump(mq, fp)
    # with open("data/rq", "wb") as fp:
    #     pickle.dump(rq, fp)
    # with open("data/sq", "wb") as fp:
    #     pickle.dump(sq, fp)

    # with open("data/mq", "rb") as fp:
    #     mq = pickle.load(fp)
    # with open("data/rq", "rb") as fp:
    #     rq = pickle.load(fp)
    # with open("data/sq", "rb") as fp:
    #     sq = pickle.load(fp)

    # with open("data/walks", "wb") as fp:
    #     pickle.dump(walks, fp)
    # with open("data/walks_200", "wb") as fp:
    #     pickle.dump(walks[:200], fp)
    # with open("scratch/data/walks_200", "rb") as fp:
    #     walks = pickle.load(fp)

    process.embedding2graph()
    
    nx_G = gv.read_graph(args)
    G = gv.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    gv.learn_embeddings(args, walks)


if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
