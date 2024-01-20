from openicl import DatasetReader, BM25Retriever, PromptTemplate, AccEvaluator, CoTInferencer
from openicl.utils.icl_common_utils import get_arabic_number, get_ref_arabic_number, strict_acc, soft_acc, shuffle_examples
from datasets import load_dataset
from models import wikiHowNet
import utils.parse_args as pa
from utils import functional
import nltk
import pickle
import pathlib
import torch


def main(args):
    # nltk.download('punkt')
    # data = DatasetReader('gsm8k', name='main', input_columns=['question'], output_column='answer')
    # template = PromptTemplate('</E> Question: </Q> \n Answer: </A>',
    #                           {'question':'</Q>', 'answer':'</A>'},
    #                           ice_token='</E>')
    # retriever = BM25Retriever(data, ice_num = 4)
    dir = pathlib.Path(__file__).resolve().parent.parent
    save_dir = dir/"scratch"/"data"
    # with open(save_dir/"retriever_3", "wb") as fp:
    #     pickle.dump(retriever, fp)
    # fp.close()
    # with open(save_dir/"retriever_1319", "rb") as fp:
    #     retriever = pickle.load(fp)
    # fp.close()
    
    # cot_list = ["Let's think step by step and put the answer after #### ."] #,
    #             # "\nTherefore, the answer (arabic number) is"]
    # inferencer = CoTInferencer(cot_list=cot_list, api_name='gpt-3.5-turbo-instruct')
    # ori_predictions = inferencer.inference(retriever, ice_template=template)
    # with open(save_dir/"predictions_gpt-3.5-turbo-instruct_sort", "wb") as fp:
    #     pickle.dump(ori_predictions, fp)
    # fp.close()
    with open(save_dir/"json_list", "rb") as fp:
        predictions = pickle.load(fp)
    fp.close()
    # predictions = get_arabic_number(predictions)
    # references = get_ref_arabic_number(data.references)
    # print(strict_acc(predictions, references))
    # print(soft_acc(predictions, references))


if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
