# import numpy as np
# import matplotlib.pyplot as plt


def preprocess(): # Three choices pca_a.txt, pca_b.txt, pca_c.txt
    with open('associationruletestdata.txt','r')as f:
        data = f.readlines()
        data_list = []
        for line in data:
            line = line.strip('\n')
            pca = line.split(sep = '\t')
            # print(pca)
            data_list.append(pca)
        return data_list

def main():
    dataset = preprocess()
    print(len(dataset[4]),len(dataset))
    print(dataset[4])

if __name__ == "__main__":
    main()