from openicl import DatasetReader, RandomRetriever, PromptTemplate, RandomRetriever, AccEvaluator, BleuEvaluator, GenInferencer, PPLInferencer
from datasets import load_dataset
from sklearn.metrics import f1_score
from models import wikiHowNet
import utils.parse_args as pa
from utils import functional
import random
import pickle
import pathlib
import torch
import time


def main(args):
    start_time = time.time()
    f1_list = []        
    acc_list = []
    test_data_random_seed = functional.random_seq(200, 20, args.random_seed)
    
    # sorting_net_work = wikiHowNet()
    dir = pathlib.Path(__file__).resolve().parent.parent
    model_dir = dir/"scratch"/"data"/"diff_sort"/"model_"
    with open(model_dir, "rb") as fp:
        sorting_net_work = pickle.load(fp)
    fp.close()
    sorting_net_work = sorting_net_work.to('cuda')
    sorting_net_work.eval()
    
    for random_seed in test_data_random_seed:
        if args.dataset == "ag_news":
            data = DatasetReader('sh0416/ag_news', input_columns=['description'], output_column='label', ds_size=128, random_seed=random_seed) 
            template = PromptTemplate(template={
                                                    1: '</E>News: </description> News type: World',
                                                    2: '</E>News: </description> News type: Sports',
                                                    3: '</E>News: </description> News type: Business',
                                                    4: '</E>News: </description> News type: Science',
                                            },
                                    column_token_map={'description' : '</description>'},
                                    ice_token='</E>'
                    )
            
        if args.dataset == "CR":
            data = DatasetReader('SetFit/CR', input_columns=['text'], output_column='label', ds_size=128, random_seed=random_seed)
            template = PromptTemplate(template={
                                                    0: '</E>Review: </text> Sentiment: negative',
                                                    1: '</E>Review: </text> Sentiment: positive',
                                            },
                                    column_token_map={'text' : '</text>'},
                                    ice_token='</E>'
                    )
            
        if args.dataset == "sst2":
            data = DatasetReader('gpt3mix/sst2', input_columns=['text'], output_column='label', ds_size=128, random_seed=random_seed)
            template = PromptTemplate(template={
                                                    0: '</E>Review: </text> Sentiment: positive',
                                                    1: '</E>Review: </text> Sentiment: negative',
                                            },
                                    column_token_map={'text' : '</text>'},
                                    ice_token='</E>'
                    )
            
        if args.dataset == "sst5":
            data = DatasetReader('SetFit/sst5', input_columns=['text'], output_column='label', ds_size=128, random_seed=random_seed)
            template = PromptTemplate(template={
                                                    0: '</E>Review: </text> Sentiment: terrible',
                                                    1: '</E>Review: </text> Sentiment: bad',
                                                    2: '</E>Review: </text> Sentiment: okay',
                                                    3: '</E>Review: </text> Sentiment: good',
                                                    4: '</E>Review: </text> Sentiment: great'
                                            },
                                    column_token_map={'text' : '</text>'},
                                    ice_token='</E>'
                    )
            
        if args.dataset == "dbpedia_14":
            data = DatasetReader('dbpedia_14', input_columns=['content'], output_column='label', ds_size=128, random_seed=random_seed)
            template = PromptTemplate(template={
                                                    0: '</E>Article: </content> Article type: Company',
                                                    1: '</E>Article: </content> Article type: School',
                                                    2: '</E>Article: </content> Article type: Artist',
                                                    3: '</E>Article: </content> Article type: Player',
                                                    4: '</E>Article: </content> Article type: Politics',
                                                    5: '</E>Article: </content> Article type: Transport',
                                                    6: '</E>Article: </content> Article type: Building',
                                                    7: '</E>Article: </content> Article type: Nature',
                                                    8: '</E>Article: </content> Article type: Village',
                                                    9: '</E>Article: </content> Article type: Animal',
                                                    10: '</E>Article: </content> Article type: Plant',
                                                    11: '</E>Article: </content> Article type: Album',
                                                    12: '</E>Article: </content> Article type: Film',
                                                    13: '</E>Article: </content> Article type: Book'
                                            },
                                    column_token_map={'content' : '</content>'},
                                    ice_token='</E>'
                    )
        
        # rand = random.Random(random_seed)
        # prompt_rand_list = list(range(128))
        # rand.shuffle(prompt_rand_list)

        retriever = RandomRetriever(data, ice_num=args.sequence_length, index_split='train', test_split='test', seed=random_seed, NP_mode=None, sst_class=args.data_classes)
        if args.model == "gpt-j-6b":
            inferencer = PPLInferencer(model_name='EleutherAI/gpt-j-6b') 
        elif args.model == 'gpt2-xl':
            inferencer = PPLInferencer(model_name='gpt2-xl') 
        elif args.model == 'NousResearch/Llama-2-7b-hf':
            inferencer = PPLInferencer(model_name='NousResearch/Llama-2-7b-hf')
        
        predictions = inferencer.inference(retriever, ice_template=template, dataset=args.dataset, sorting_net_work=sorting_net_work, shuffle_mode=args.shuffle_mode, llm=args.model)
        print(predictions)
        print(data.references)
        acc_score = AccEvaluator().score(predictions=predictions, references=data.references)
        print("Acc: ",acc_score['accuracy'])
        acc_list.append(acc_score['accuracy'])
        f1score = f1_score(predictions, data.references, average='macro')
        f1_list.append(f1score)
        print("F1: ", f1score)
        
    end_time = time.time()
    duration = end_time - start_time
    print("it took "+str(duration)+" seconds for "+args.model+"to sort 20x128 cases with "+str(args.sequence_length)+" ices in "+args.dataset)
    f1_avg = sum(f1_list)/len(f1_list)
    acc_avg = sum(acc_list)/len(acc_list)
    print("the average f1 is: "+str(f1_avg))
    print("the average acc is: " + str(acc_avg))


if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
