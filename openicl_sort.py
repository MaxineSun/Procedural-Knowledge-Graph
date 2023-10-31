from openicl import DatasetReader, RandomRetriever, PromptTemplate, RandomRetriever, PPLInferencer, ZeroRetriever, GenInferencer, AccEvaluator
from models import wikiHowNet
import utils.parse_args as pa
import pickle
import torch


def main(args):
    # Define a DatasetReader, loading dataset from huggingface and selecting 5 pieces of data randomly.
    data = DatasetReader('gpt3mix/sst2', input_columns=['text'], output_column='label', ds_size=8)

    # SST-2 Template Example
    template = PromptTemplate(template={
                                            0: '</E>Positive Movie Review: </text>',
                                            1: '</E>Negative Movie Review: </text>'
                                    },
                            column_token_map={'text' : '</text>'},
                            ice_token='</E>'
            )
    # TopK Retriever
    retriever = RandomRetriever(data, ice_num=4, index_split='train', test_split='test', seed=2, NP_mode='NP')
    # Define a Inferencer
    inferencer = PPLInferencer(model_name='distilgpt2')
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
    

if __name__ == "__main__":
    args = pa.parse_args()
    main(args)