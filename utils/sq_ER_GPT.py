import openai
import csv
import numpy as np
import pickle
import time



class GPT_Val_Process:
    def __init__(self):
        self.sq_raw = []

    def set_pnn_examples(self):
        # set positive test examples
        with open("../scratch/data/sq_cls05", "rb") as fp:
            sq_cls = pickle.load(fp)
        fp.close()
        u_sq_cls = np.unique(sq_cls)
        u_len = len(u_sq_cls)
        comp1 = np.empty(shape=0, dtype=int)
        comp2 = np.empty(shape=0, dtype=int)
        print(u_len)
        for i in range(u_len):
            if len(comp1) >= 500:
                break
            sublist = np.where(sq_cls == i)[0]
            l = len(sublist)
            if l == 1:
                continue
            elif l >= 5:
                comp1 = np.concatenate((comp1, sublist[:5]), axis=0)
                comp2 = np.concatenate((comp2, sublist[-5:][::-1]), axis=0)
            else:
                comp1 = np.concatenate((comp1, sublist), axis=0)
                comp2 = np.concatenate((comp2, sublist[::-1]), axis=0)
        with open("../scratch/data/comp1", "wb") as fp:
            pickle.dump(comp1, fp)
        fp.close()
        with open("../scratch/data/comp2", "wb") as fp:
            pickle.dump(comp2, fp)
        fp.close()

        # set negative examples
        comn1 = np.empty(shape=0, dtype=int)
        comn2 = np.empty(shape=0, dtype=int)
        for i in reversed(range(u_len)):
            if len(comn1) >= 500:
                break
            sublist = np.where(sq_cls == i)[0]
            l = len(sublist)
            if l == 1:
                obj = sublist[0]
                comn1 = np.concatenate(
                    (comn1, np.array([obj, obj, obj, obj, obj])), axis=0
                )
                comn2 = np.concatenate(
                    (comn2, np.random.randint(0, obj - 1, size=5)), axis=0
                )
            elif l >= 5:
                comn1 = np.concatenate((comn1, sublist[:5]), axis=0)
                lower = np.min(sublist)
                comn2 = np.concatenate(
                    (comn2, np.random.randint(0, lower - 1, size=5)), axis=0
                )
            else:
                comn1 = np.concatenate((comn1, sublist), axis=0)
                lower = np.min(sublist)
                sub_l = len(sublist)
                comn2 = np.concatenate(
                    (comn2, np.random.randint(0, lower - 1, size=sub_l)), axis=0
                )
        with open("../scratch/data/comn1", "wb") as fp:
            pickle.dump(comn1, fp)
        fp.close()
        with open("../scratch/data/comn2", "wb") as fp:
            pickle.dump(comn2, fp)
        fp.close()

        return

    def gen_manual_examples(self):
        with open("../scratch/data/sqlist.csv", newline="") as csvfile:
            self.sq_raw = list(csv.reader(csvfile, delimiter=","))

        advise = self.sq_raw[95991]
        advise.append(self.sq_raw[13084][0])
        advise.append(self.sq_raw[14157][0])
        advise.append(self.sq_raw[152621][0])
        advise.append(self.sq_raw[118720][0])

        fit = self.sq_raw[219960]
        fit.append(self.sq_raw[37001][0])
        fit.append(self.sq_raw[68984][0])

        diaphragm = self.sq_raw[75643]
        diaphragm.append(self.sq_raw[267005][0])
        diaphragm.append(self.sq_raw[40079][0])
        diaphragm.append(self.sq_raw[118844][0])
        diaphragm.append(self.sq_raw[228541][0])

        practice = self.sq_raw[5565]
        practice.append(self.sq_raw[5291][0])
        practice.append(self.sq_raw[256][0])
        practice.append(self.sq_raw[1210][0])
        practice.append(self.sq_raw[4800][0])
        practice.append(self.sq_raw[4680][0])

        tapok = self.sq_raw[55544]
        tapok.append(self.sq_raw[63134][0])
        tapok.append(self.sq_raw[63218][0])
        tapok.append(self.sq_raw[160639][0])
        tapok.append(self.sq_raw[176408][0])

        doctor = self.sq_raw[55804]
        doctor.append(self.sq_raw[57971][0])
        doctor.append(self.sq_raw[58218][0])
        doctor.append(self.sq_raw[58489][0])
        doctor.append(self.sq_raw[58664][0])
        doctor.append(self.sq_raw[59069][0])
        doctor.append(self.sq_raw[59145][0])
        doctor.append(self.sq_raw[61092][0])

        topic_list = [advise, diaphragm, practice, tapok]  # , doctor , ]
        prompt = "The task is to find similiar entities. For "
        for topic in topic_list:
            for item in topic:
                prompt = prompt + '"' + item + '", '
            prompt += "they have the same meaning. For "
        list_len = len(topic_list)
        for i in range(list_len):
            for j in range(i + 1, list_len):
                prompt += (
                    '"'
                    + topic_list[i][0]
                    + '" and "'
                    + topic_list[j][0]
                    + '", they have different meanings. '
                )
        return prompt

    def gpt_val(self):
        # self.set_pnn_examples()
        # prompt = self.gen_manual_examples()

        # with open("../scratch/data/comp1", "rb") as fp:
        #     comp1 = pickle.load(fp)
        # fp.close()

        # with open("../scratch/data/comp2", "rb") as fp:
        #     comp2 = pickle.load(fp)
        # fp.close()

        # with open("../scratch/data/comn1", "rb") as fp:
        #     comn1 = pickle.load(fp)
        # fp.close()

        # with open("../scratch/data/comn2", "rb") as fp:
        #     comn2 = pickle.load(fp)
        # fp.close()

        # openai.api_key = "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"

        # GPT validation
        # l = len(comp1)
        # l = 100
        # scorep = np.zeros(l)
        # for i in range(10):
        #     first = comp1[i]
        #     second = comp2[i]
        #     quest_prompt = (
        #         prompt
        #         + 'Now for "'
        #         + self.sq_raw[first][0]
        #         + '" and "'
        #         + self.sq_raw[second][0]
        #         + '", do they have the same or different meanings?'
        #     )
        #     response = openai.Completion.create(
        #         engine="text-ada-001",
        #         prompt=quest_prompt,
        #         max_tokens=10,
        #         n=1,
        #         stream=False,
        #         stop=None,
        #         temperature=1.0,
        #     )
        #     if (
        #         "same" in response["choices"][0]["text"]
        #         or "Same" in response["choices"][0]["text"]
        #     ):
        #         scorep[i] = 0
        #     if "ifferent" in response["choices"][0]["text"]:
        #         scorep[i] = 1
        # with open("../scratch/data/scorep", "wb") as fp:
        #     pickle.dump(scorep, fp)
        # fp.close()

        # # l = len(comn1)
        # l = 100
        # scoren = np.ones(l)
        # for i in range(l):
        #     first = comn1[i]
        #     second = comn2[i]
        #     quest_prompt = (
        #         prompt
        #         + 'Now for "'
        #         + self.sq_raw[first][0]
        #         + '" and "'
        #         + self.sq_raw[second][0]
        #         + '", do they have the same or different meanings?'
        #     )
        #     response = openai.Completion.create(
        #         engine="text-ada-001",
        #         prompt=quest_prompt,
        #         max_tokens=10,
        #         n=1,
        #         stream=False,
        #         stop=None,
        #         temperature=1.0,
        #     )
        #     if (
        #         "same" in response["choices"][0]["text"]
        #         or "Same" in response["choices"][0]["text"]
        #     ):
        #         scoren[i] = 0
        #     if "ifferent" in response["choices"][0]["text"]:
        #         scoren[i] = 1
        # with open("../scratch/data/scoren", "wb") as fp:
        #     pickle.dump(scoren, fp)
        # fp.close()

        with open("../scratch/data/scoren", "rb") as fp:
            scoren = pickle.load(fp)
        fp.close()

        with open("../scratch/data/scorep", "rb") as fp:
            scorep = pickle.load(fp)
        fp.close()

        with open("../scratch/data/comn1", "rb") as fp:
            comn1 = pickle.load(fp)[:100]
        fp.close()

        with open("../scratch/data/comn2", "rb") as fp:
            comn2 = pickle.load(fp)[:100]
        fp.close()

        with open("../scratch/data/comp1", "rb") as fp:
            comp1 = pickle.load(fp)[:100]
        fp.close()

        with open("../scratch/data/comp2", "rb") as fp:
            comp2 = pickle.load(fp)[:100]
        fp.close()

        with open("../scratch/data/sq_cls06", "rb") as fp:
            sq_cls = pickle.load(fp)
        fp.close()

        precn = 0.0
        for ind, (f,s) in enumerate(zip(comn1, comn2)):
            if (sq_cls[s]==sq_cls[f]) and (scoren[ind]==0):
                precn += 1
            elif (sq_cls[s]!=sq_cls[f]) and (scoren[ind]==1):
                precn += 1

        precp = 0.0
        for ind, (f,s) in enumerate(zip(comp1, comp2)):
            if (sq_cls[s]==sq_cls[f]) and (scorep[ind]==0):
                precp += 1
            elif (sq_cls[s]!=sq_cls[f]) and (scorep[ind]==1):
                precp += 1

        print("precision of negative samples is: " + str(precn / 100))
        print("precision of positive samples is: " + str(precp / 100))

        return
