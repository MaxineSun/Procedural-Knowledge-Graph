import pickle
import numpy as np
import utils.parse_args as pa
import matplotlib.pyplot as plt
import pathlib


def main(args):
    dir = pathlib.Path(__file__).resolve().parent.parent
    dir = dir/"scratch"/"data"/"diff_sort"/"plot_100"
    if args.dataset == "sst5" and args.model == 'gpt2-xl':
        with open(dir/"ori_5_gpt2-xl_8"/"entropy", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(dir/"ori_5_gpt2-xl_8"/"f1_list", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_5_gpt2-xl_8"/"entropy", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_5_gpt2-xl_8"/"f1_list", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        with open(dir/"bubble_5_gpt2-xl_8"/"entropy", "rb") as fp:
            x_bubble = pickle.load(fp)
        fp.close()
        with open(dir/"bubble_5_gpt2-xl_8"/"f1_list", "rb") as fp:
            y_bubble = pickle.load(fp)
        fp.close()
        y_ori = [item for item in y_ori]
        y_sorted = [item for item in y_sorted]
        y_bubble = [item for item in y_bubble]
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        plot_bubble(x_bubble, y_bubble)
        plt.title('sst5_gpt2-xl')
        plt.xlabel('entropy')
        plt.ylabel('F1')
        plt.legend(fontsize=6)
        plt.savefig('sst5_gpt2-xl_8.png')
        
    if args.dataset == "sst5" and args.model == 'EleutherAI/gpt-j-6b':
        with open(dir/"ori_5_gpt-j-6b_16"/"entropy", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(dir/"ori_5_gpt-j-6b_16"/"score_list", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_5_gpt-j-6b_16"/"entropy", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_5_gpt-j-6b_16"/"score_list", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        # with open(dir/"bubble_5_gpt-j-6b_16"/"entropy", "rb") as fp:
        #     x_bubble = pickle.load(fp)
        # fp.close()
        # with open(dir/"bubble_5_gpt-j-6b_16"/"score_list", "rb") as fp:
        #     y_bubble = pickle.load(fp)
        # fp.close()
        plt.figure(figsize=(15, 6))
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        # plot_bubble(x_bubble, y_bubble)
        plt.title('sst5_gpt-j-6b')
        plt.xlabel('entropy')
        plt.ylabel('accuracy')
        plt.legend(fontsize=6)
        plt.savefig('sst5_gpt-j-6b_100.png')
    
    if args.dataset == "sst5" and args.model == 'NousResearch/Llama-2-7b-hf':
        with open(dir/"ori_5_Llama-2-7b-hf_16"/"entropy", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(dir/"ori_5_Llama-2-7b-hf_16"/"score_list", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_5_Llama-2-7b-hf_16"/"entropy", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_5_Llama-2-7b-hf_16"/"score_list", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        # with open(dir/"bubble_5_gpt-j-6b_16"/"entropy", "rb") as fp:
        #     x_bubble = pickle.load(fp)
        # fp.close()
        # with open(dir/"bubble_5_gpt-j-6b_16"/"score_list", "rb") as fp:
        #     y_bubble = pickle.load(fp)
        # fp.close()
        plt.figure(figsize=(15, 6))
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        # plot_bubble(x_bubble, y_bubble)
        plt.title('sst5_Llama-2-7b-hf')
        plt.xlabel('entropy')
        plt.ylabel('accuracy')
        plt.legend(fontsize=6)
        plt.savefig('sst5_Llama-2-7b-hf_100.png')
        
    if args.dataset == "sst2" and args.model == 'gpt2-xl':
        
        with open(dir/"ori_2_gpt2-xl_16"/"entropy", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(dir/"ori_2_gpt2-xl_16"/"f1_list", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_2_gpt2-xl_16"/"entropy", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_2_gpt2-xl_16"/"f1_list", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        with open(dir/"bubble_2_gpt2-xl_16"/"entropy", "rb") as fp:
            x_bubble = pickle.load(fp)
        fp.close()
        with open(dir/"bubble_2_gpt2-xl_16"/"f1_list", "rb") as fp:
            y_bubble = pickle.load(fp)
        fp.close()
        y_ori = [item for item in y_ori]
        y_sorted = [item for item in y_sorted]
        y_bubble = [item for item in y_bubble]
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        plot_bubble(x_bubble, y_bubble)
        plt.title('sst2_gpt2-xl')
        plt.xlabel('entropy')
        plt.ylabel('F1')
        plt.legend(fontsize=6)
        plt.savefig('sst2_gpt2-xl_16.png')
        
    if args.dataset == "sst2" and args.model == 'Llama-2-7b-hf':
        
        with open(dir/"ori_2_Llama-2-7b-hf_16"/"entropy", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(dir/"ori_2_Llama-2-7b-hf_16"/"f1_list", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_2_Llama-2-7b-hf_16"/"entropy", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(dir/"sorted_2_Llama-2-7b-hf_16"/"f1_list", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        with open(dir/"bubble_2_Llama-2-7b-hf_16"/"entropy", "rb") as fp:
            x_bubble = pickle.load(fp)
        fp.close()
        with open(dir/"bubble_2_Llama-2-7b-hf_16"/"f1_list", "rb") as fp:
            y_bubble = pickle.load(fp)
        fp.close()
        y_ori = [item for item in y_ori]
        y_sorted = [item for item in y_sorted]
        y_bubble = [item for item in y_bubble]
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        plot_bubble(x_bubble, y_bubble)
        plt.title('sst2_Llama-2-7b-h')
        plt.xlabel('entropy')
        plt.ylabel('F1')
        plt.legend(fontsize=6)
        plt.savefig('sst2_Llama-2-7b-h_16.png')
        
    if args.dataset == "sst2" and args.model == 'EleutherAI/gpt-j-6b':
        with open(f"../scratch/data/diff_sort/plot/entropy_ori_2_E", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/plot/score_list_ori_2_E", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/plot/entropy_sorted_2_E", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/plot/score_list_sorted_2_E", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/plot/entropy_bubble_2_E", "rb") as fp:
            x_bubble = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/plot/score_list_bubble_2_E", "rb") as fp:
            y_bubble = pickle.load(fp)
        fp.close()
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        plot_bubble(x_bubble, y_bubble)
        plt.title('sst2_EleutherAI/gpt-j-6b')
        plt.xlabel('entropy')
        plt.ylabel('accuracy')
        plt.legend(fontsize=6)
        plt.savefig('sst2_gpt-j-6b.png')


def plot_ori(x, y):
    # plt.scatter(x, y, marker='.', color='#a29988', label='Ori Scatter Plot')
    x_label = list(set(x))
    y_mean = []
    y_upper = []
    y_lower = []
    for item in x_label:
        idx = [xi==item for xi in x]
        tmp_y = [element for element, condition in zip(y, idx) if condition]
        tmp_mean = np.mean(tmp_y)
        tmp_std = np.std(tmp_y)/2
        y_mean.append(tmp_mean)
        y_upper.append(tmp_mean + tmp_std)
        y_lower.append(tmp_mean - tmp_std)
    plt.plot(x_label, y_mean, marker='.', linestyle='-', linewidth=1, color='#a29988', label='Ori Mean Curve')
    # plt.plot(x_label, y_upper, marker='.', linestyle='-', linewidth=1, color='#cac3bb', label='Ori Upper Curve')
    # plt.plot(x_label, y_lower, marker='.', linestyle='-', linewidth=1, color='#cac3bb', label='Ori Lower Curve')
    plt.fill_between(x_label, y_upper, y_lower, color='#dadad8', alpha=0.3)
    
def plot_sort(x, y):
    # plt.scatter(x, y, marker='.', color='#965454', label='Sorted Scatter Plot')
    x_label = list(set(x))
    y_mean = []
    y_upper = []
    y_lower = []
    for item in x_label:
        idx = [xi==item for xi in x]
        tmp_y = [element for element, condition in zip(y, idx) if condition]
        tmp_mean = np.mean(tmp_y)
        tmp_std = np.std(tmp_y)/2
        y_mean.append(tmp_mean)
        y_upper.append(tmp_mean + tmp_std)
        y_lower.append(tmp_mean - tmp_std)
    plt.plot(x_label, y_mean, marker='.', linestyle='-', linewidth=1, color='#965454', label='Sorted Mean Curve')
    # plt.plot(x_label, y_upper, marker='.', linestyle='-', linewidth=1, color='#a27e7e', label='Sorted Upper Curve')
    # plt.plot(x_label, y_lower, marker='.', linestyle='-', linewidth=1, color='#a27e7e', label='Sorted Lower Curve')
    plt.fill_between(x_label, y_upper, y_lower, color='#ead0d1', alpha=0.3)
    
def plot_bubble(x, y):
    # plt.scatter(x, y, marker='.', color='#8696a7', label='Reverse Scatter Plot')
    x_label = list(set(x))
    y_mean = []
    y_upper = []
    y_lower = []
    for item in x_label:
        idx = [xi==item for xi in x]
        tmp_y = [element for element, condition in zip(y, idx) if condition]
        tmp_mean = np.mean(tmp_y)
        tmp_std = np.std(tmp_y)/2
        y_mean.append(tmp_mean)
        y_upper.append(tmp_mean + tmp_std)
        y_lower.append(tmp_mean - tmp_std)
    plt.plot(x_label, y_mean, marker='.', linestyle='-', linewidth=1, color='#8696a7', label='Bubble Mean Curve')
    # plt.plot(x_label, y_upper, marker='.', linestyle='-', linewidth=1, color='#9ca8b8', label='Reverse Lower Curve')
    # plt.plot(x_label, y_lower, marker='.', linestyle='-', linewidth=1, color='#9ca8b8', label='Reverse Lower Curve')
    plt.fill_between(x_label, y_upper, y_lower, color='#c1cbd7', alpha=0.3)
    
    

if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
