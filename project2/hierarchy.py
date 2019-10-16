import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint

def dist_cal(data):
    width,length = data.shape
    matrix = np.zeros((width,width))
    for i in range(width):
        for j in range(i,width):
            if i == j:
                matrix[i][j] = 0
            else:
                norm2 = np.linalg.norm(data[i] - data[j])
                matrix[i][j] = matrix[j][i] = norm2
    return matrix

def min_find(data,min_index):
    min_value = data[0][1]
    length = data.shape[0]
    for row in range(length):
        for col in range(row,length):
            if row == col:
                break
            elif data[row][col]< min_value:
                min_value = data[row][col]
                min_index[0] = row
                min_index[1] = col

    return min_index



def matrix_update(old, index, id):
    a = index[0]
    b = index[1]
    for i in range(old.shape[0]):
        old[a][i] = min(old[a][i], old[b][i])
        old[i][a] = min(old[i][a], old[i][b])
    up_id = (id[a],id[b])
    id[a] = list(up_id)
    old = np.delete(old, b, axis = 0)
    old = np.delete(old, b, axis = 1)

    del id[b]
    return old, id

def merge_list(input):
    res = []
    if type(input) == list:
        for i in input:
            res  = res + merge_list(i)
    else:
        res.append(input)
    return res


def classify_and_plot(label, x_axis,y_axis,name):
    category1 = pd.Categorical(label).categories
    category_len = len(category1)
    for i in category1:
        if i == -1:
            category_len = len(category1)-1


    data_group = [[] for i in range(category_len)]
    outlier = []
    for i in range(category_len):
        index2 = 0
        for j in label:
            if j == -1:
                outlier.append(index2)
            if i == j:
                data_group[i].append(index2)
            index2 += 1
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'fuchsia','yellow','cyan','lime']

    if outlier:
        groupx = []
        groupy = []
        for i in outlier:
            groupx.append(x_axis[i])
            groupy.append(y_axis[i])
        plt.scatter(groupx, groupy, c = 'k', marker='.')
    index3 = 0
    for i in data_group:
        group_x = []
        group_y = []
        for j in i:
            group_y.append(y_axis[j])
            group_x.append(x_axis[j])
        plt.scatter(group_x, group_y, c= color[index3], marker='.')
        index3+=1

    plt.legend(labels=category1, loc='upper right')
    plt.title(name)
    # plt.savefig(name+'.eps')
    plt.show()


def main():
    # file = "iyer.txt"
    file = "cho.txt"

    data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 2:])

    ground_truth = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 1])

    id = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 0])

    X = (data - data.mean(0))

    k = 5

    # print(len(id))
    #initialize the matrix and ip list
    dist_matrix = dist_cal(data)
    update_matrix = dist_matrix
    min_index = [1,0]
    new_id = id

    #initialize the minimum distance points


    while(len(new_id)>k):
        min_index = min_find(update_matrix,min_index)
        update_matrix, new_id = matrix_update(update_matrix, min_index, new_id)

    # print(update_matrix, new_id)


    merged_res = []
    for i in new_id:
        merged_res.append(merge_list(i))

    cov = np.cov(X.T)

    eg_value,eg_vector = np.linalg.eig(cov)
    # print(eg_value)
    idx = eg_value.argsort()[::-1]
    eg_value = eg_value[idx]
    eg_vector = eg_vector[:, idx]
    rank1 = np.dot(X,eg_vector[:,0]).flatten()
    rank2 = np.dot(X,eg_vector[:,1]).flatten()

    # classify_and_plot(ground_truth, rank1, rank2, 'PCA_result')
    count = 0
    gen_result = [0 for i in range(len(ground_truth))]
    for i in merged_res:
        count+=1
        for j in i:
            gen_result[j-1] = count

    classify_and_plot(gen_result, rank1, rank2, 'PCA_HAG_result')



if __name__ == "__main__":
    main()