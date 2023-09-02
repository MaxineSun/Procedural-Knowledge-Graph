import argparse


def parse_args():
    # Parses the node2vec arguments.
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--jsonpath', nargs='?', default="../scratch/data/wikihow.json", help='Input json file path')
    parser.add_argument('--datasetpath', nargs='?', default="../scratch/data/", help='dataset file path')
    parser.add_argument('--input', nargs='?', default='../scratch/data/graph/sq.edgelist', help='Input graph path')
    parser.add_argument('--output', nargs='?', default='../scratch/data/emb/wikihow.emb', help='Embeddings path')
    parser.add_argument('--modelpath', nargs='?', default='../scratch/data/model/wikihow.model', help='Models path')
    parser.add_argument('--sq_hash_path', nargs='?', default='../scratch/data/sq_hash', help='sq hash table path')
    parser.add_argument('--train_model', nargs='?', default="random guess", help='baseline or gnn')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=50, help='Length of walk per source. Default is 50.')
    parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=10, type=int, help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1000, help='Return hyperparameter. Default is 1000.')
    parser.add_argument('--density', type=float, default=0.65, help='dbscan density.')
    parser.add_argument('--q', type=float, default=0.3, help='Inout hyperparameter. Default is 0.3.')
    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.') 
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)
    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.add_argument('-m', '--method', type=str, default='odd_even', choices=['odd_even', 'bitonic'])
    parser.add_argument('-n', '--num_compare', type=int, default=8)
    parser.add_argument('-s', '--steepness', type=float, default=10)
    parser.add_argument('-a', '--art_lambda', type=float, default=0.25)
    parser.add_argument('-accu', '--NUM_ACCUMULATION_STEPS', type=int, default=1024)
    parser.add_argument('--nsg_batch_size', type=int, default=16, help='batch size of next sentence generation model, default is 32.')
    parser.add_argument('--learning_rate_nsg', type=float, default=5e-7, help='learning rate of next sentence generation model, default is 0.001.')
    parser.add_argument('--lr_MLP', type=float, default=1e-4)
    parser.add_argument('--lr_encoder', type=float, default=1e-7)
    parser.set_defaults(directed=False)

    return parser.parse_args()
