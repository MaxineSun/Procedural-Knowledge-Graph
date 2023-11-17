from openicl import DatasetReader, RandomRetriever, PromptTemplate, RandomRetriever, ZeroRetriever, AccEvaluator, BleuEvaluator, GenInferencer, PPLInferencer
from datasets import load_dataset
from models import wikiHowNet
import utils.parse_args as pa
from utils import functional
import pickle
import torch


def main(args):
    if args.case == "bi-class":
        seqs = []
        entropy = []
        score_list = []
        count = [0]*args.sequence_length
        if args.dataset == "sst2":
            data = DatasetReader('gpt3mix/sst2', input_columns=['text'], output_column='label', ds_size=128)
            template = PromptTemplate(template={
                                                    0: '</E>Positive Movie Review: </text>',
                                                    1: '</E>Negative Movie Review: </text>',
                                            },
                                    column_token_map={'text' : '</text>'},
                                    ice_token='</E>'
                    )
        
        if args.dataset == "sst5":
            data = DatasetReader('SetFit/sst5', input_columns=['text'], output_column='label', ds_size=128)
            template = PromptTemplate(template={
                                                    0: '</E>Very negative Movie Review: </text>',
                                                    1: '</E>Negative Movie Review: </text>',
                                                    2: '</E>Neutral Movie Review: </text>',
                                                    3: '</E>Positive Movie Review: </text>',
                                                    4: '</E>Very positive Movie Review: </text>'
                                            },
                                    column_token_map={'text' : '</text>'},
                                    ice_token='</E>'
                    )

        model = wikiHowNet()
        with open("../scratch/data/diff_sort/model", "rb") as fp:
            model = pickle.load(fp)
        fp.close()
        model = model.to('cuda')
        model.eval()
        
        while len(entropy)<90:
            seq = functional.random_seq(args.data_classes, args.sequence_length)
            tmp_entropy = functional.seq_entropy(seq)
            if count[tmp_entropy]<11 and tmp_entropy>args.data_classes:
                count[tmp_entropy] += 1
                seqs.append(seq)
                entropy.append(functional.seq_entropy(seq))
        
        for seq in seqs:
            args.NP_mode = seq
            retriever = RandomRetriever(data, ice_num=args.sequence_length, index_split='train', test_split='test', seed=args.random_seed, NP_mode=args.NP_mode, sst_class=args.data_classes)
            inferencer = PPLInferencer(model_name=args.model) 
            predictions = inferencer.inference(retriever, ice_template=template, output_json_filename='sst5', sorting_net_work=model, shuffle_mode=args.shuffle_mode)
            score = AccEvaluator().score(predictions=predictions, references=data.references)
            score_list.append(score['accuracy'])
        
            
        if args.dataset == "sst5" and args.model == 'EleutherAI/gpt-j-6b' and args.shuffle_mode == 'ori':
            with open("../scratch/data/diff_sort/entropy_ori_5_E", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_ori_5_E", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst5" and args.model == 'EleutherAI/gpt-j-6b' and args.shuffle_mode == 'sorted':
            with open("../scratch/data/diff_sort/entropy_sorted_5_E", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_sorted_5_E", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst5" and args.model == 'EleutherAI/gpt-j-6b' and args.shuffle_mode == 'reverse':
            with open("../scratch/data/diff_sort/entropy_reverse_5_E", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_reverse_5_E", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst5" and args.model == 'gpt2-xl' and args.shuffle_mode == 'ori':
            with open("../scratch/data/diff_sort/entropy_ori_5_g", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_ori_5_g", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst5" and args.model == 'gpt2-xl' and args.shuffle_mode == 'sorted':
            with open("../scratch/data/diff_sort/entropy_sorted_5_g", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_sorted_5_g", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst5" and args.model == 'gpt2-xl' and args.shuffle_mode == 'reverse':
            with open("../scratch/data/diff_sort/entropy_reverse_5_g", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_reverse_5_g", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst2" and args.model == 'EleutherAI/gpt-j-6b' and args.shuffle_mode == 'ori':
            with open("../scratch/data/diff_sort/entropy_ori_2_E_8", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_ori_2_E_8", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst2" and args.model == 'EleutherAI/gpt-j-6b' and args.shuffle_mode == 'sorted':
            with open("../scratch/data/diff_sort/entropy_sorted_2_E_8", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_sorted_2_E_8", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst2" and args.model == 'EleutherAI/gpt-j-6b' and args.shuffle_mode == 'reverse':
            with open("../scratch/data/diff_sort/entropy_reverse_2_E_8", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_reverse_2_E_8", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst2" and args.model == 'gpt2-xl' and args.shuffle_mode == 'ori':
            with open("../scratch/data/diff_sort/entropy_ori_2_g_8", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_ori_2_g_8", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst2" and args.model == 'gpt2-xl' and args.shuffle_mode == 'sorted':
            with open("../scratch/data/diff_sort/entropy_sorted_2_g_8", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_sorted_2_g_8", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        elif args.dataset == "sst2" and args.model == 'gpt2-xl' and args.shuffle_mode == 'reverse':
            with open("../scratch/data/diff_sort/entropy_reverse_2_g_8", "wb") as fp:
                pickle.dump(entropy, fp)
            fp.close()
            with open("../scratch/data/diff_sort/score_list_reverse_2_g_8", "wb") as fp:
                pickle.dump(score_list, fp)
            fp.close()
        
    if args.case == "translate":
        dataset = load_dataset("wmt16", 'de-en').map(lambda example: example['translation'])
        data = DatasetReader(dataset, input_columns=['de'], output_column='en')
        template = PromptTemplate('</E> German:</German> \n English: </English>',{'de':'</German>', 'en':'</English>'}, ice_token='</E>')
        retriever = RandomRetriever(data, ice_num=16, index_split='train', test_split='test', seed=2, NP_mode=args.NP_mode)
        inferencer = GenInferencer(model_name='facebook/xglm-7.5B')
        predictions = inferencer.inference(retriever, ice_template=template)
        score = BleuEvaluator().score(predictions=predictions, references=data.references)
        print(score)
        

if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
