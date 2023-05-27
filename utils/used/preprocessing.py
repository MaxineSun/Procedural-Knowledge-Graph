import os
import re
import json
from tqdm import tqdm


def clean_string(s):
    s = re.sub(r'http\S+', '', s)  # remove url
    s = re.sub(r'<.*?>', '', s)  # remove html code (object)
    s = re.sub(r'{.*?}', '', s)  # remove html code (header)
    s = re.sub('[^A-Za-z0-9 ]+\'', '', s)  # remove special character
    s = re.sub('\t+', '\t', s)  # remove repeat \t
    s = re.sub('\n+', '\n', s)  # remove repeat \n
    s = re.sub(' +', ' ', s)  # remove repeat ' '
    return s


def preprocess_wikihow(raw_data_path, process_data_path):
    # get all raw data files
    files_name = []
    for root, dirs, files in os.walk(raw_data_path):
        files_name = files
    # process files
    data = []
    # get necessary k-v pairs
    key1_list = ['title_description', 'title']
    key2_list = ['category_hierarchy', 'related_articles', 'QAs']
    key3_list = ['steps', 'parts', 'methods']
    check_num = 0
    for f_name in tqdm(files_name):
        f_path = os.path.join(raw_data_path, f_name)
        with open(f_path, "r") as f:
            d = json.loads(f.read())
            tmp_d = {}
            check = 1  # ignore item with missing features
            for k, v in d.items():
                if k in key1_list:
                    if v is None:
                        check = 0
                        continue
                    tmp_d[k] = clean_string(v)
                elif k in key2_list:
                    if v is None:
                        check = 0
                        continue
                    if k == "category_hierarchy":
                        tmp_d[k] = [clean_string(t) for t in v]
                    elif k == "related_articles":
                        tmp_d[k] = [clean_string(t["title"]) for t in v]
                    else:  # QAs
                        tmp_d[k] = []
                        for tmp_l in v:
                            if tmp_l[0] is None or tmp_l[1] is None:
                                check = 0
                                continue
                            tmp_d[k].append({"question": clean_string(tmp_l[0]), "answer": clean_string(tmp_l[1])})
                elif k in key3_list:  # steps, parts, methods
                    tmp_d[k] = v
            if check:
                data.append(tmp_d)
            else:
                check_num += 1
    print("The number of ignored item is: ", check_num)
    # process steps, parts, methods
    data_new = []
    count_step, count_part, count_method = 0, 0, 0
    for d in tqdm(data):
        len_check = []
        for k in key3_list:
            if k in d:
                len_check.append(len(d[k]))
                if len(d[k]) > 0:
                    if k == "steps":
                        count_step += 1
                    elif k == "parts":
                        count_part += 1
                    else:
                        count_method += 1
            else:
                len_check.append(0)
        len_check = sorted(len_check)
        if len_check[1] != 0:  # multiple key?
            print("Warning, multiple keys from [steps, parts, methods]!")
            continue
        if len_check[2] == 0:  # multiple key?
            print("Warning, all keys from [steps, parts, methods] are empty!")
            continue

        d_new = {}
        # main question
        d_new["main_question"] = d["title"].capitalize() + "?"
        d_new["main_description"] = d["title_description"]
        # related questions
        d_new["related_questions"] = []
        if "related_articles" in d:
            for related_q in d["related_articles"]:
                d_new["related_questions"].append(related_q.capitalize() + "?")
        if "QAs" in d:
            for related_qa in d["QAs"]:
                d_new["related_questions"].append(related_qa["question"].capitalize() + "?")
        # subquestion and descriptions
        d_new["sub_questions"] = []
        d_new["sub_descriptions"] = []
        sub_key = ""
        if "steps" in d:
            sub_key = "steps"
        elif "parts" in d:
            sub_key = "parts"
        elif "methods" in d:
            sub_key = "methods"
        if sub_key == "steps":
            tmp_sub = []
            tmp_des = []
            for step in d[sub_key]:
                tmp_sub.append(clean_string(step["headline"]))
                tmp_des.append(clean_string(step["description"]))
            tmp_sub = [ts for ts in tmp_sub if len(ts) > 16]
            tmp_des = [td for td in tmp_des if len(td) > 16]
            if len(tmp_sub) >= 1 and len(tmp_des) >= 1:
                # if len(tmp_sub) > 1 and len(tmp_sub) == len(tmp_des) and "" not in tmp_sub and "" not in tmp_des:
                d_new["sub_questions"].append(tmp_sub)
                d_new["sub_descriptions"].append(tmp_des)
        else:
            for steps in d[sub_key]:
                tmp_sub = []
                tmp_des = []
                for step in steps['steps']:
                    tmp_sub.append(clean_string(step["headline"]))
                    tmp_des.append(clean_string(step["description"]))
                tmp_sub = [ts for ts in tmp_sub if len(ts) > 16]
                tmp_des = [td for td in tmp_des if len(td) > 16]
                if len(tmp_sub) >= 1 and len(tmp_des) >= 1:
                    # if len(tmp_sub) > 1 and len(tmp_sub) == len(tmp_des) and "" not in tmp_sub and "" not in tmp_des:
                    d_new["sub_questions"].append(tmp_sub)
                    d_new["sub_descriptions"].append(tmp_des)
        # check d_new, skip if something is missing
        if len(d_new["main_question"]) == 0: continue
        if len(d_new["main_description"]) == 0: continue
        if len(d_new["related_questions"]) == 0: continue
        if len(d_new["sub_questions"]) == 0: continue
        if len(d_new["sub_descriptions"]) == 0: continue
        data_new.append(d_new)
    print("The number of steps, parts, methods, processed data number are: ", count_step, count_part, count_method,
          len(data_new))
    # write down the data
    with open(process_data_path, "w") as f:
        for d in data_new:
            f.write(json.dumps(d))
            f.write("\n")

    return


if __name__ == "__main__":
    raw_data_path = "../js_files_en"
    process_data_path = "../wikihow_1.json"
    preprocess_wikihow(raw_data_path, process_data_path)
