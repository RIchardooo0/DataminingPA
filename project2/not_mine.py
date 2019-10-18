import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

def incidence_mat_gen(label):
    matrix = [[0]*len(label) for i in range(len(label))]
    for i in range(len(label)):
        for j in range(i,len(label)):
            if label[i] == label[j]:
                matrix[i][j] = 1
                matrix[j][i] = 1
            else:
                matrix[i][j] = 0
                matrix[j][i] = 0
    return np.array(matrix)

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
        for col in range(row,length-1):
            if row == col:
                continue
            elif data[row][col]< min_value:
                min_value = data[row][col]
                print(row, col)
                min_index[0] = row
                min_index[1] = col

    return min_index

# file = "cho.txt"
#
# data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 2:])
#
# ground_truth = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 1])
#
# id = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 0])
#
# X = (data - data.mean(0))
#
# # k = input("Please input # clusters:\n")
# k = 5
#
# # print(len(id))
# #initialize the matrix and ip list
# dist_matrix = dist_cal(data)
# update_matrix = dist_matrix
# min_index = [1,0]
# new_id = id

a = [1,1,1,2,2]
ground = [1,1,2,2,2]
a_in = incidence_mat_gen(a)
ground_in = incidence_mat_gen(ground)
print(a_in)
print(ground_in)
M11 = np.sum(ground_in * a_in)

new_inci = ground_in + a_in
count = 0
for i in range(len(new_inci)):
    for j in range(len(new_inci[0])):
        if new_inci[i][j] == 0:
            count += 1
print(len(ground_in))
M00 = count
rand = (M11 + M00) / (len(ground_in) ** 2)
jaccard = M11 / (len(ground_in) ** 2 - M00)
print(rand, jaccard)