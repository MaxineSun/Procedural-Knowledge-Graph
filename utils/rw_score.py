import torch
import math
import random
from tqdm import tqdm
import pickle
from itertools import combinations
from transformers import BertTokenizer
from torch.utils.data import Dataset


class ParisDataset(Dataset):
    def __init__(self, pairs, score):
        self.pairs = pairs
        self.score = score

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.score[idx]

class Data_Process:
    def __init__(self):
        with open("../scratch/data/json_list", "rb") as fp:
            self.json_list = pickle.load(fp)
        fp.close()
    
    def dist_full(self, dist):
        for item in dist:
            if item <2:
                return False
        return True
    
    def ed4fullconnected(self, num_nodes):
        node_combinations = list(combinations(range(num_nodes), 2))
        edge_index = torch.tensor(node_combinations, dtype=torch.long).t().contiguous()
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        
        return edge_index

    def get_subgraph(self, mq_ind):
        sub_graph = [self.json_list[mq_ind]["main_question"]]
        for sq_item in self.json_list[mq_ind]["sub_questions"][0]:
            if len(sub_graph) <7:
                sub_graph.append(sq_item)
        # num_nodes = len(sub_graph)
        # edge_index = self.ed4fullconnected(num_nodes)
        # weight = []
        # with open("../scratch/data/model/model15", "rb") as fp:
        #     model = pickle.load(fp)
        # fp.close()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model_name = 'bert-base-uncased'
        # tokenizer = BertTokenizer.from_pretrained(model_name)
        # for i in range(len(edge_index[0])):
        #     encoded_inputs = tokenizer(sub_graph[edge_index[0][i]], sub_graph[edge_index[1][i]], return_tensors='pt', padding=True, truncation=True).to(device)
        #     outputs = model(**encoded_inputs).logits
        #     weight.append(outputs[:,0])
        # weight = torch.tensor(weight)
        # mask = weight> 6.0
        # edge_index = edge_index[:,mask]
        # weight = weight[mask]

        return sub_graph #, edge_index, weight
    
    def random_walk(self, length, edge_index = None, weight = None):
        walk = [0]
        pre = 0
        if edge_index is None:
            edge_index = self.ed4fullconnected(length)
        if weight is None:
            num_nodes = int(edge_index.max()) + 1
            for _ in range(length-1):
                neighbors = edge_index[1][edge_index[0] == walk[-1]]
                neighbors = neighbors.tolist()
                neighbors = [x for x in neighbors if x not in walk]
                if len(neighbors) == 0:
                    next_node = random.randint(0, num_nodes - 1)
                next_node = random.choice(neighbors)
                walk.append(next_node)
        if weight is not None:
                num_nodes = int(edge_index.max()) + 1
                for step in range(length-1):
                    current_node = walk[-1]
                    masks = edge_index[0] == current_node
                    masks = masks.tolist()
                    masks = [False if x in walk else mask for mask,x in zip(masks, edge_index[1])]
                    if sum(masks) == 0:
                        next_node = random.randint(0, num_nodes - 1)
                    else:
                        neighbor_weights = weight[masks]
                        neighbor_probs = neighbor_weights / neighbor_weights.sum()
                        next_node = torch.multinomial(neighbor_probs, 1).item()
                    walk.append(next_node)
        return walk

    def score(self, walk):
        score = 0
        walk_len = len(walk)
        if walk == list(range(walk_len)):
            return 1
        else:
            # Unsortedp
            score+=1
        for i in range(walk_len):
            for j in range(i + 1, walk_len):
                # inversions
                if walk[i] > walk[j]:
                    score += 1
            # adjacent inversions
            if i<walk_len-1:
                if walk[i] > walk[i+1]:
                    score +=1
        # insertion index
        lengths = [1] * walk_len
        for i in range(1, walk_len):
            for j in range(i):
                if walk[i] > walk[j]:
                    lengths[i] = max(lengths[i], lengths[j] + 1)
        score += walk_len
        score -= max(lengths)
        norm = walk_len*walk_len/2+5*walk_len/2-2
        return 1-(score/norm)
    
    def get_pairs(self, subgraph, walk):
        input = []
        target = []
        for i in range(len(walk)-1):
            input.append(subgraph[walk[i]])
            target.append(subgraph[walk[i+1]])
        return input, target, self.score(walk)
    
    def gen_data(self):
        pairs = []
        scores = []
        for i in tqdm(range(10000)): # range(37000)
            sub_graph = self.get_subgraph(i)
            num_nodes = len(sub_graph)
            if num_nodes<7:
                continue
            dist = [0]*10
            dist[-1] = 1
            # correct order walk
            walk = list(range(num_nodes))
            input, target, score = self.get_pairs(sub_graph, walk)
            # random walk
            walks = [walk]
            while not self.dist_full(dist) and sum(dist)<7:
                walk = self.random_walk(num_nodes)
                score = self.score(walk)
                if dist[int(10*(score-0.001))]<2:
                    walks.append(walk)
                    dist[int(10*score-0.001)] += 1
            if len(walks) == 7:
                for walk in walks:
                    input, target, score = self.get_pairs(sub_graph, walk)
                    pairs.append([input, target])
                    scores.append(score)
        data = ParisDataset(pairs, scores)
        with open(f"../scratch/data/rw/random_walk_data1k", "wb") as fp:
            pickle.dump(data, fp)
        fp.close()
        return data

