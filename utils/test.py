import jsonlines
from tqdm import tqdm


class Data_Process:
    def __init__(self):
        self.mqlen = 0

    def test(self, filepath):
        json_list = []
        with open(filepath, "r+") as f_in:
            for json in jsonlines.Reader(f_in):
                # if len(json_list) < 40:
                json_list.append(json)
        mq_raw = [item["main_question"] for item in json_list]
        all_rq_count = 0
        new_rq_count = 0
        for ind, item in tqdm(enumerate(json_list)):
            for rq_item in item["related_questions"]:
                all_rq_count +=1
                if rq_item not in mq_raw:
                    new_rq_count +=1
            if ind%100 ==0:
                print("all rq "+str(all_rq_count)+", and new rq"+str(new_rq_count))