'''Random Retriever'''

from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from tqdm import trange
import numpy as np
import utils.parse_args as pa
from accelerate import Accelerator

logger = get_logger(__name__)


class RandomRetriever(BaseRetriever):
    """Random In-context Learning Retriever Class
        Class of Random Retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        seed (`int`, optional): Seed for the random number generator.
    """

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 seed: Optional[int] = 43,
                 accelerator: Optional[Accelerator] = None,
                 NP_mode = None,
                 sst_class = 2
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.seed = seed
        self.NP_mode = NP_mode
        self.sst_class = sst_class
        self.dataset_reader = DatasetReader._check_dataset_reader(dataset_reader)

    def retrieve(self):
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        args = pa.parse_args()
        
        
        if self.NP_mode == "NP":
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 1:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            return rtr_idx_list
        
        elif type(self.NP_mode) is list:
            label_list = ['Agent', 'Place', 'Species', 'Work', 'Event', 'UnitOfWork', 'TopicalConcept', 'Device', 'SportsSeason']
            if self.ice_num != len(self.NP_mode):
                raise ValueError("The index number is not correct.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                for i in self.NP_mode:
                    while len(idx_list) < self.ice_num:
                        idx = int(np.random.choice(num_idx, 1))
                        # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                        # print(label_list[i])
                        if args.dataset in ["sst5", "sst2"]:
                            if self.dataset_reader.dataset["train"][idx][dr.output_column] == i:
                                idx_list.append(idx)
                                break
                        if args.dataset == "DeveloperOats/DBPedia_Classes":
                            if self.dataset_reader.dataset["train"][idx][dr.output_column] == label_list[i]:
                                idx_list.append(idx)
                                break
                rtr_idx_list.append(idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "NNPP":
            if self.ice_num != 4:
                raise ValueError("The index number is not 4.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "NPNP":
            if self.ice_num != 4:
                raise ValueError("The index number is not 4.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 3:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num -2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num -1:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "NPPN":
            if self.ice_num != 4:
                raise ValueError("The index number is not 4.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 3:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num -1:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "PNNP":
            if self.ice_num != 4:
                raise ValueError("The index number is not 4.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 3:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num -1:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "PNPN":
            if self.ice_num != 4:
                raise ValueError("The index number is not 4.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 3:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num -2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num -1:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "PPNN":
            if self.ice_num != 4:
                raise ValueError("The index number is not 4.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "PPPPNNNN":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 4:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "NNNNPPPP":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 4:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "NNPPNNPP":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 6:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 4:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "PPNNPPNN":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 6:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 4:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list

        elif self.NP_mode == "PPNNNNPP":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 6:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "NNPPPPNN":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 6:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "PNNNNPPP":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 7:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 3:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "NPPPPNNN":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 7:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 3:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        elif self.NP_mode == "NPNPNPNP":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 7:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 6:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 5:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 4:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 3:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 1:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx) 
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)        
                
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list

        elif self.NP_mode == "PNPNPNPN":
            if self.ice_num != 8:
                raise ValueError("The index number is not 8.")
            dr = self.dataset_reader
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = []
                while len(idx_list) < self.ice_num - 7:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 6:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 5:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 4:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 3:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 2:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)
                while len(idx_list) < self.ice_num - 1:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 0:
                        idx_list.append(idx) 
                while len(idx_list) < self.ice_num:
                    idx = int(np.random.choice(num_idx, 1))
                    # print(self.dataset_reader.dataset["train"][idx][dr.output_column])
                    if self.dataset_reader.dataset["train"][idx][dr.output_column] == 1:
                        idx_list.append(idx)        
                rtr_idx_list.append(idx_list)
            # print(rtr_idx_list)
            return rtr_idx_list
        
        else:
            for _ in trange(len(self.test_ds), disable=not self.is_main_process):
                idx_list = np.random.choice(num_idx, self.ice_num, replace=False).tolist()
                rtr_idx_list.append(idx_list)
            return rtr_idx_list
        
        
