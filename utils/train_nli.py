import pickle
import torch
import math
import torch.optim as optim
from datetime import datetime
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import utils.parse_args as pa
import pathlib
import time


def shift_list(lst, a):
    n = len(lst)
    a = a % n  
    shifted_lst = lst[a:] + lst[:a]
    return shifted_lst

def get_samples(json_list):
    input_head = [item["main_question"] for item in json_list]
    target_head = [item["sub_questions"][0][0] for item in json_list]
    y_head = [1.0]*len(input_head)
    y_head += [0.0]*len(input_head)
    input_head = input_head*2
    target_head = target_head + shift_list(target_head, 4)

    samples = []
    for inp, tar, lab in zip(input_head, target_head, y_head):
        samples.append(InputExample(texts=[inp, tar], label=lab))
        samples.append(InputExample(texts=[tar, inp], label=lab))

    input_follow = []
    target_follow = []
    for item in json_list:
        for ind, sq in enumerate(item["sub_questions"][0][:-1]):
            input_follow.append(sq)
            next_item = item["sub_questions"][0][ind+1]
            target_follow.append(next_item)

    y_follow = [1.0]*len(input_follow)
    y_follow += [0.5]*len(input_follow)
    y_follow += [0.0]*len(input_follow)
    input_follow = input_follow*3
    target_follow = target_follow + shift_list(target_follow, 4) + shift_list(target_follow, 24)
    for inp, tar, lab in zip(input_follow, target_follow, y_follow):
        samples.append(InputExample(texts=[inp, tar], label=lab))
        samples.append(InputExample(texts=[tar, inp], label=lab))
    return samples

def score(walk):
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

def test_seq():
    with open("../scratch/data/json_list", "rb") as fp:
        json_list = pickle.load(fp)
    fp.close()
    data_len = len(json_list)
    train_len = int(0.6*data_len)
    val_len = int(0.2*data_len)
    test_list = json_list[train_len+val_len:]

    model_save_path = '../scratch/data/nlisentence-transformers-all-mpnet-base-v2-2023-07-29_12-23-46'
    model = SentenceTransformer(model_save_path)

    score_list = []
    for ind_mq, item in enumerate(test_list):
        test_order = [0]
        # mq to sq1
        mq = item["main_question"]
        mq_emb = model.encode(mq)
        sq_list = [mq]+item["sub_questions"][0]
        max_ind = 0
        max_sim = 0.0
        for ind_sq, sq in enumerate(sq_list[1:], start=1):
            sq_emb = model.encode(sq)
            sim = paired_cosine_distances(mq_emb.reshape(1,-1), sq_emb.reshape(1,-1))[0]
            if sim >max_sim:
                max_sim = sim
                max_ind = ind_sq
        test_order.append(max_ind)

        # sq1 to the end
        while(len(test_order)!=len(sq_list)):
            max_ind = 0
            max_sim = 0.0
            sq1 = sq_list[test_order[-1]]
            sq_emb1 = model.encode(sq1)
            for ind_sq2, sq2 in enumerate(sq_list[1:], start=1):
                if ind_sq2 != test_order[-1] and ind_sq2 not in test_order:
                    sq_emb2 = model.encode(sq2)
                    sim = paired_cosine_distances(sq_emb1.reshape(1,-1), sq_emb2.reshape(1,-1))[0]
                    if sim > max_sim:
                        max_sim = sim
                        max_ind = ind_sq2
            test_order.append(max_ind)
        score_list.append(score(test_order))
        print(test_order)
    print(score_list)
    print(sum(score_list)/len(score_list))

def data_split(args):
    with open("../scratch/data/json_list", "rb") as fp:
        json_list = pickle.load(fp)
    fp.close()
    # json_list = json_list[:8000]
    data_len = len(json_list)
    train_len = int(0.6*data_len)
    val_len = int(0.2*data_len)
    train_list = json_list[:train_len]
    val_list = json_list[train_len:train_len+val_len]
    test_list = json_list[train_len+val_len:]
    train_samples = get_samples(train_list)
    val_samples = get_samples(val_list)
    test_samples = get_samples(test_list)
    return train_samples, val_samples, test_samples

def train(args):
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    train_batch_size = 128
    max_seq_length = 75
    num_epochs = 1

    model_save_path = '../scratch/data/nli'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # model_save_path = '../scratch/data/nlisentence-transformers-all-mpnet-base-v2-2023-07-29_12-23-46'
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=torch.nn.Tanh())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    train_samples, val_samples, test_samples = data_split(args)
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1)
    val_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_samples, name='sts-dev')
    model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=val_evaluator, epochs=1, warmup_steps=warmup_steps, output_path=model_save_path, use_amp=False)

    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
    test_evaluator(model) # , output_path=model_save_path)
    print(test_evaluator(model))

def sort_bub():
    start_time = time.time()
    dir = pathlib.Path(__file__).resolve().parent.parent
    for i in range(55):
        file_name = f"sq_set{str(i).zfill(5)}"
        sq_dir = dir/"scratch"/"data"/"diff_sort"/"separate_sq_set"/file_name
        with open(sq_dir, "rb") as fp:
            sq_set_shuffled = pickle.load(fp)
        fp.close()
        
        file_name = f"target_idx{str(i).zfill(5)}"
        ti_dir = dir/"scratch"/"data"/"diff_sort"/"separate_target_idx"/file_name
        with open(ti_dir, "rb") as fp:
            target_idx_set = pickle.load(fp)
        fp.close()
        
        
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Bubble sort uses {elapsed_time} seconds")
        
        
        
        

if __name__ == "__main__":
    # args = pa.parse_args()
    # train(args)
    sort_bub()