import pickle
import numpy as np
import utils.parse_args as pa
import matplotlib.pyplot as plt


def main(args):
    if args.dataset == "sst5" and args.model == 'gpt2-xl':
        with open(f"../scratch/data/diff_sort/entropy_ori_5_g", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_ori_5_g", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/entropy_sorted_5_g", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_sorted_5_g", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/entropy_reverse_5_g", "rb") as fp:
            x_reverse = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_reverse_5_g", "rb") as fp:
            y_reverse = pickle.load(fp)
        fp.close()
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        plot_reverse(x_reverse, y_reverse)
        plt.title('sst5_gpt2-xl')
        plt.xlabel('entropy')
        plt.ylabel('accuracy')
        plt.legend(fontsize=6)
        plt.savefig('sst5_gpt2-xl.png')
        
    if args.dataset == "sst5" and args.model == 'EleutherAI/gpt-j-6b':
        with open(f"../scratch/data/diff_sort/entropy_ori_5_E", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_ori_5_E", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/entropy_sorted_5_E", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_sorted_5_E", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/entropy_reverse_5_E", "rb") as fp:
            x_reverse = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_reverse_5_E", "rb") as fp:
            y_reverse = pickle.load(fp)
        fp.close()
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        plot_reverse(x_reverse, y_reverse)
        plt.title('sst5_EleutherAI/gpt-j-6b')
        plt.xlabel('entropy')
        plt.ylabel('accuracy')
        plt.legend(fontsize=6)
        plt.savefig('sst5_gpt-j-6b.png')
        
    if args.dataset == "sst2" and args.model == 'gpt2-xl':
        with open(f"../scratch/data/diff_sort/entropy_ori_2_g", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_ori_2_g", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/entropy_sorted_2_g", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_sorted_2_g", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/entropy_reverse_2_g", "rb") as fp:
            x_reverse = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_reverse_2_g", "rb") as fp:
            y_reverse = pickle.load(fp)
        fp.close()
        print(x_ori)
        print(y_ori)
        print(x_sorted)
        print(y_sorted)
        print(x_reverse)
        print(y_reverse)
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        plot_reverse(x_reverse, y_reverse)
        plt.title('sst2_gpt2-xl')
        plt.xlabel('entropy')
        plt.ylabel('accuracy')
        plt.legend(fontsize=6)
        plt.savefig('sst2_gpt2-xl.png')
        
    if args.dataset == "sst2" and args.model == 'EleutherAI/gpt-j-6b':
        with open(f"../scratch/data/diff_sort/entropy_ori_2_E", "rb") as fp:
            x_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_ori_2_E", "rb") as fp:
            y_ori = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/entropy_sorted_2_E", "rb") as fp:
            x_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_sorted_2_E", "rb") as fp:
            y_sorted = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/entropy_reverse_2_E", "rb") as fp:
            x_reverse = pickle.load(fp)
        fp.close()
        with open(f"../scratch/data/diff_sort/score_list_reverse_2_E", "rb") as fp:
            y_reverse = pickle.load(fp)
        fp.close()
        plot_ori(x_ori, y_ori)
        plot_sort(x_sorted, y_sorted)
        plot_reverse(x_reverse, y_reverse)
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
    
def plot_reverse(x, y):
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
    plt.plot(x_label, y_mean, marker='.', linestyle='-', linewidth=1, color='#8696a7', label='Reverse Mean Curve')
    # plt.plot(x_label, y_upper, marker='.', linestyle='-', linewidth=1, color='#9ca8b8', label='Reverse Lower Curve')
    # plt.plot(x_label, y_lower, marker='.', linestyle='-', linewidth=1, color='#9ca8b8', label='Reverse Lower Curve')
    plt.fill_between(x_label, y_upper, y_lower, color='#c1cbd7', alpha=0.3)
    
    

if __name__ == "__main__":
    args = pa.parse_args()
    main(args)
