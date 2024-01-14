from openicl import DatasetReader, RandomRetriever, PromptTemplate, RandomRetriever, AccEvaluator, BleuEvaluator, GenInferencer, PPLInferencer
from datasets import load_dataset
from sklearn.metrics import f1_score
from models import wikiHowNet
import utils.parse_args as pa
from utils import functional
import pickle
import pathlib
import torch
import time


def main(args):
    start_time = time.time()
    if args.case == "bi-class":
        # seqs = []
        # entropy = []
        
        score_list = []
        count = [0]*args.sequence_length*args.data_classes
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
        dir = pathlib.Path(__file__).resolve().parent.parent
        model_dir = dir/"scratch"/"data"/"diff_sort"/"model_"
        with open(model_dir, "rb") as fp:
            model = pickle.load(fp)
        fp.close()
        model = model.to('cuda')
        model.eval()
        
        # while len(entropy)<(100*(args.sequence_length-args.data_classes)*(args.data_classes-1)):
        #     seq = functional.random_seq(args.data_classes, args.sequence_length)
        #     tmp_entropy = functional.seq_entropy(seq)
        #     if count[tmp_entropy]<100 and tmp_entropy>=args.data_classes:
        #         count[tmp_entropy] += 1
        #         seqs.append(seq)
        #         entropy.append(tmp_entropy)
        
        # print(entropy)
        # print(len(entropy))
        # print(count)
        # print(seqs[0])
        # with open("../scratch/data/diff_sort/entropy_"+str(args.sequence_length)+"_"+str(args.data_classes), "wb") as fp:
        #     pickle.dump(entropy, fp)
        # fp.close()
        
        # with open("../scratch/data/diff_sort/seqs_"+str(args.sequence_length)+"_"+str(args.data_classes), "wb") as fp:
        #     pickle.dump(seqs, fp)
        # fp.close()
        # return
        
        entropy_name = "entropy_"+str(args.sequence_length)+"_"+str(args.data_classes)
        entropy_dir = dir/"scratch"/"data"/"diff_sort"/entropy_name
        with open(entropy_dir, "rb") as fp:
            entropy = pickle.load(fp)
            entropy = entropy[50*args.parallel_id:50*(args.parallel_id+1)]
        fp.close()
        seqs_name = "seqs_"+str(args.sequence_length)+"_"+str(args.data_classes)
        seqs_dir = dir/"scratch"/"data"/"diff_sort"/seqs_name
        with open(seqs_dir, "rb") as fp:
            seqs = pickle.load(fp)
            seqs = seqs[50*args.parallel_id:50*(args.parallel_id+1)]
        fp.close()
        
        for seq in seqs:
            args.NP_mode = seq
            retriever = RandomRetriever(data, ice_num=args.sequence_length, index_split='train', test_split='test', seed=args.random_seed, NP_mode=args.NP_mode, sst_class=args.data_classes)
            if args.model == "gpt-j-6b":
                inferencer = PPLInferencer(model_name='EleutherAI/gpt-j-6b') 
            elif args.model == 'gpt2-xl':
                inferencer = PPLInferencer(model_name='gpt2-xl') 
            elif args.model == 'NousResearch/Llama-2-7b-hf':
                inferencer = PPLInferencer(model_name=args.model)
            predictions = inferencer.inference(retriever, ice_template=template, output_json_filename='sst', sorting_net_work=model, shuffle_mode=args.shuffle_mode)
            score = AccEvaluator().score(predictions=predictions, references=data.references)
            # score = f1_score(predictions, data.references, average='macro')
            print(score)
            score_list.append(score)
        
        end_time = time.time()
        duration = end_time - start_time
        print("it took "+str(duration)+" seconds for "+args.model+"to sort 50 cases with "+str(args.sequence_length)+" ices in "+args.dataset)
            
        dir_name = args.shuffle_mode+"_"+str(args.data_classes)+"_"+args.model+'_'+str(args.sequence_length)
        save_dir = dir/"scratch"/"data"/"diff_sort"/"plot_100"/dir_name
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            
        entropy_file_name = "entropy_"+str(args.parallel_id)
        with open(save_dir/entropy_file_name, "wb") as fp:
            pickle.dump(entropy, fp)
        fp.close()
        
        score_list_file_name = "score_list_"+str(args.parallel_id)
        with open(save_dir/score_list_file_name, "wb") as fp:
            pickle.dump(score_list, fp)
        fp.close()
        
    # if args.case == "translate":
    #     dataset = load_dataset("wmt16", 'de-en').map(lambda example: example['translation'])
    #     data = DatasetReader(dataset, input_columns=['de'], output_column='en')
    #     template = PromptTemplate('</E> German:</German> \n English: </English>',{'de':'</German>', 'en':'</English>'}, ice_token='</E>')
    #     retriever = RandomRetriever(data, ice_num=16, index_split='train', test_split='test', seed=2, NP_mode=args.NP_mode)
    #     inferencer = GenInferencer(model_name='facebook/xglm-7.5B')
    #     predictions = inferencer.inference(retriever, ice_template=template)
    #     score = BleuEvaluator().score(predictions=predictions, references=data.references)
    #     print(score)
        

if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
