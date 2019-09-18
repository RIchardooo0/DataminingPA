from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


def preprocess(string): # Three choices pca_a.txt, pca_b.txt, pca_c.txt
    with open(string,'r')as f:
        data = f.readlines()
        data_list = []
        for line in data:
            line = line.strip('\n')
            pca = line.split(sep="\t")
            data_list.append(pca)
        return data_list



def main():
    dataset1 = preprocess('pca_a.txt')
    dataset2 = preprocess('pca_b.txt')
    dataset3 = preprocess('pca_c.txt')
    data_1 = pd.dataframe(dataset1)
    dataMat = np.array(data_1)

    pca_tsne = TSNE(n_components=2)
    newMat = pca_tsne.fit_transform(dataMat)


if __name__ == "__main__":
    main()
