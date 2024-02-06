import pickle 
import pathlib
import utils.parse_args as pa

def main(args):
    dir = pathlib.Path(__file__).resolve().parent.parent
    dir = dir/"scratch"/"data"/"diff_sort"/"plot_100"/args.data_dir
    entropy = "entropy"
    entropy_list = []
    for file_name in sorted(dir.iterdir(), key=lambda x: int(x.stem.split('_')[-1])):
        if file_name.is_file() and file_name.name.startswith(f"{entropy}_"):
            print(file_name)
            with open(file_name, "rb") as fp:
                tmp_list = pickle.load(fp)
            fp.close()
            entropy_list += tmp_list
    
    acc = "acc_list"
    acc_list = []
    for file_name in sorted(dir.iterdir(), key=lambda x: int(x.stem.split('_')[-1])):
        if file_name.is_file() and file_name.name.startswith(f"{acc}_"):
            print(file_name)
            with open(file_name, "rb") as fp:
                tmp_list = pickle.load(fp)
            fp.close()
            acc_list += tmp_list
            
    f1 = "f1_list"
    f1_list = []
    for file_name in sorted(dir.iterdir(), key=lambda x: int(x.stem.split('_')[-1])):
        if file_name.is_file() and file_name.name.startswith(f"{f1}_"):
            print(file_name)
            with open(file_name, "rb") as fp:
                tmp_list = pickle.load(fp)
            fp.close()
            f1_list += tmp_list
            
    with open(dir/"entropy", "wb") as fp:
        pickle.dump(entropy_list, fp)
    fp.close()
    
    with open(dir/"acc_list", "wb") as fp:
        pickle.dump(acc_list, fp)
    fp.close()
    
    with open(dir/"f1_list", "wb") as fp:
        pickle.dump(f1_list, fp)
    fp.close()


if __name__ == "__main__":
    args = pa.parse_args()
    main(args)