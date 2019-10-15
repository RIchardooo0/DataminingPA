import numpy as np
import matplotlib as plt
import pandas as pd


file = "iyer.txt"
# file2 = "cho.txt"

data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:,2:])

ground_truth =list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:,1])

id = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:,0])

X = (data - data.mean(0))

k = 10

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

def min_find(data):
    data[data == 0] = 100
    index = np.where(data == np.min(data))
    data[data == 100] = 0
    return index[0]



def matrix_update(old, index, id):
    a = index[0]
    b = index[1]

    for i in range(old.shape[0]):
        old[a][i] = min (old[a][i], old[b][i])
        old[i][a] = min (old[i][a], old[i][b])
    up_id = (id[a],id[b])
    id[a] = list(up_id)
    old = np.delete(old, b, axis = 0)
    old = np.delete(old, b, axis = 1)

    del id[b]

    return old, id


print(len(id))
#initialize the matrix and ip list
dist_matrix = dist_cal(data)
update_matrix = dist_matrix
new_id = id
while(len(new_id)>k):
    index = min_find(update_matrix)
    update_matrix, new_id = matrix_update(update_matrix, index, new_id)

print(update_matrix, new_id)
