from sklearn import manifold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess(string): # Three choices pca_a.txt, pca_b.txt, pca_c.txt
    with open(string,'r')as f:
        data = f.readlines()
        data_list = []
        for line in data:
            line = line.strip('\n')
            pca = line.split(sep="\t")
            data_list.append(pca)
        return data_list

def classify_and_plot(label, x_axis,y_axis):
    category1 = pd.Categorical(label).categories
    cat_mapping = {}
    index1 = 0
    for i in category1:
        cat_mapping[i] = index1
        index1 += 1

    new = pd.Series(label).map(cat_mapping)
    data_group = [[] for i in range(len(category1))]

    for i in range(len(category1)):
        index2 = 0
        for j in new:
            if i == j:
                data_group[i].append(index2)
            index2 += 1

    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    index3 = 0
    for i in data_group:
        group_x = []
        group_y = []
        for j in i:
            group_y.append(y_axis[j])
            group_x.append(x_axis[j])
        plt.scatter(group_x, group_y, c=color[index3], marker='.')


        index3 += 1
    plt.legend(labels=category1, loc='upper right')
    plt.show()

def main():
    dataset1 = preprocess('pca_a.txt')
    dataset2 = preprocess('pca_b.txt')
    dataset3 = preprocess('pca_c.txt')
    data = np.array(dataset1)
    label = data[:, -1]
    num_data = data[:, :-1]
    num_data = np.array(num_data).astype(np.float)

    pca_tsne = manifold.TSNE(n_components=2,init = 'pca')
    pca_tsne.fit_transform(num_data)
    newMat = pca_tsne.embedding_.T

    classify_and_plot(label,newMat[0],newMat[1])


if __name__ == "__main__":
    main()
