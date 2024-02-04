import statistics


def main():
    with open('sorted_ag_4_gpt-j-6b_16.out','r') as file:
        list_content = [line.strip() for line in file]
    file.close()

    out = []
    for item in list_content:
        if item[:3]=='F1:':
            out.append(item)
    
    f1list = []
    for item in out:
        f1list.append(str2float(item))
    
    # print(f1list)
    # ref = []
    # for i in [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]:
    #     ref += str2list(out[i])
    
    print(statistics.stdev(f1list))
    
    return statistics.stdev(f1list)
    
def str2float(f1str):
    return float(f1str[5:])
    
if __name__ == "__main__":
    main()
