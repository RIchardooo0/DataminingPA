import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def preprocess(string): # Three choices pca_a.txt, pca_b.txt, pca_c.txt
    with open(string,'r')as f:
        data = f.readlines()
        data_list = []
        for line in data:
            pca = line.split()
            data_list.append(pca)
        return data_list

def svd_plot(data):
    data_n = np.array(data)
    attribute_num = len(data_n[0]) - 1
    datanew = np.matrix(data_n[:,:attribute_num],dtype = 'float')
    u, s, vt = np.linalg.svd(datanew)
    print(s)

def main():
    dataset1 = preprocess('pca_a.txt')
    dataset2 = preprocess('pca_b.txt')
    dataset3 = preprocess('pca_c.txt')
    svd_plot(dataset1)

if __name__ == "__main__":
    main()


