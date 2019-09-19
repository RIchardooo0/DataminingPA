import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd


def preprocess(): # Three choices pca_a.txt, pca_b.txt, pca_c.txt
    with open('associationruletestdata.txt','r')as f:
        data = f.readlines()
        data_list = []
        for line in data:
            line = line.strip('\n')
            pca = line.split(sep="\t")
            data_list.append(pca)
        return data_list

def set_generation(itemset, length, int):
    list1 = []
    for i in combinations(itemset, int):
        if len(list(set(i[0]+i[1]))) == length:
            a = list(set(i[0] + i[1]))
            # list1.append(list(set(i[0]+i[1])))
            list1.append(a)

    return list1

def main():
    Data = preprocess()
    Dat = pd.DataFrame(Data)


    for i in range(len(Dat.columns) - 1):
        Dat[i] = 'G' + str(i + 1) + "_" + Dat[i].astype(str)
    Data_set = Dat
    support = input("Please input the support\t")
    single_candidate = set()


    for i in range(len(Dat.columns)):
        dat_col = Dat[i].groupby(Dat[i]).describe()
        for j in range(len(dat_col)):
            count = list(dat_col.iloc[j])[2:4]
            if (count[1] >= int(support)):
                single_candidate.add(count[0])
    single_candidate = [[i] for i in single_candidate]
    print('number of length-1 frequent itemsets:\n'+ str(len(single_candidate)))
    data_list = []

    for i in range(len(Data)):
        data_list.append(Data_set.iloc[i])
    next_level = single_candidate

    for length in range(2,len(Dat.columns)):
        lis = set_generation(next_level,length,2)
        print(lis)
        next_level = []
        for i in lis:
            counter = 0
            for j in data_list:
                if set(i).issubset(j):
                    counter+=1
            if counter>=int(support):
                next_level.append(i)
        if len(next_level)==0:
            print("No more rules")
            break
        print('number of length-'+str(length)+'\tfrequent itemsets:\t'+ str(len(next_level)))






    # for comb in range(len(Data.columns)):
    #     lis = set_generation(single_candidate, comb)
    #     for pair in lis:
    #


if __name__ == "__main__":
    main()