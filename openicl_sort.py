from openicl import DatasetReader, RandomRetriever, PromptTemplate, RandomRetriever, PPLInferencer, ZeroRetriever, GenInferencer, AccEvaluator, BleuEvaluator
from datasets import load_dataset
from models import wikiHowNet
import utils.parse_args as pa
import pickle
import torch


def main(args):
    if args.case == "bi-class":
        # Define a DatasetReader, loading dataset from huggingface and selecting 5 pieces of data randomly.
        data = DatasetReader('gpt3mix/sst2', input_columns=['text'], output_column='label', ds_size=128)

        # SST-2 Template Example
        template = PromptTemplate(template={
                                                0: '</E>Positive Movie Review: </text>',
                                                1: '</E>Negative Movie Review: </text>'
                                        },
                                column_token_map={'text' : '</text>'},
                                ice_token='</E>'
                )
        # TopK Retriever
        retriever = RandomRetriever(data, ice_num=4, index_split='train', test_split='test', seed=2, NP_mode=args.NP_mode)
        # Define a Inferencer
        inferencer = PPLInferencer(model_name='gpt2-xl')
        # # Inference
        # model = wikiHowNet()
        with open("../scratch/data/diff_sort/model", "rb") as fp:
            model = pickle.load(fp)
        fp.close()
        model = model.to('cuda')
        model.eval()
        predictions = inferencer.inference(retriever, ice_template=template, output_json_filename='sst2', sorting_net_work=model, shuffle_mode=args.shuffle_mode)
        score = AccEvaluator().score(predictions=predictions, references=data.references)
        print(score)
        
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
