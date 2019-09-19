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
# def set_generation(itemset, length):

def main():
    Data = preprocess()
    Data = pd.DataFrame(Data)


    for i in range(len(Data.columns) - 1):
        Data[i] = 'G' + str(i + 1) + "_" + Data[i].astype(str)

    support = input("Please input the support\t")
    single_candidate = set()


    for i in range(len(Data.columns)):
        dat_col = Data[i].groupby(Data[i]).describe()
        for j in range(len(dat_col)):
            count = list(dat_col.iloc[j])[2:4]
            if (count[1] >= int(support)):
                single_candidate.add(count[0])
    print(single_candidate)



if __name__ == "__main__":
    main()