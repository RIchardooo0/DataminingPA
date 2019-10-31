import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import sys


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
        for col in range(row,length):
            if row == col:
                continue
            elif data[row][col]<= min_value:
                min_value = data[row][col]
                min_index[0] = row
                min_index[1] = col

    return min_index



def matrix_update(update, index, id):
    a = index[0]
    b = index[1]
    for i in range(update.shape[0]):
        update[a][i] = min(update[a][i], update[b][i])
        update[i][a] = min(update[i][a], update[i][b])
    up_id = (id[a],id[b])
    id[a] = list(up_id)
    update = np.delete(update, b, axis = 0)
    update = np.delete(update, b, axis = 1)

    del id[b]

    return update, id

def merge_list(input):
    res = []
    if type(input) == list:
        for i in input:
            res  = res + merge_list(i)
    else:
        res.append(input)
    return res
def ja_rand_cal(truth, result):
    M11 = np.sum(truth*result)

    new_inci = truth + result
    count = 0
    for i in range(len(new_inci)):
        for j in range(len(new_inci[0])):
            if new_inci[i][j] == 0:
                count+=1
    M00 =count
    rand = (M11+M00)/(len(truth)**2)
    jaccard = M11/(len(truth)**2 - M00)

    return rand, jaccard


def main():
    # file = "iyer.txt"
    # file = "cho.txt"
    file = sys.argv[1]

    data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 2:])

    ground_truth = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 1])

    id = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 0])

    k = sys.argv[2]

    # print(len(id))
    #initialize the matrix and ip list
    dist_matrix = dist_cal(data)
    update_matrix = dist_matrix
    min_index = [0,1]
    new_id = id
    #initialize the minimum distance points

    while(len(new_id)>int(k)):
        min_index = min_find(update_matrix,min_index)
        update_matrix, new_id = matrix_update(update_matrix, min_index, new_id)


    merged_res = []
    for i in new_id:
        merged_res.append(merge_list(i))

    count = 0
    gen_result = [0 for i in range(len(ground_truth))]
    for i in merged_res:
        count+=1
        for j in i:
            gen_result[j-1] = count

    inci_truth = incidence_mat_gen(ground_truth)
    inci_hier = incidence_mat_gen(gen_result)

    rand, jaccard = ja_rand_cal(inci_truth, inci_hier)
    print(rand, jaccard)
    print(new_id)
#PCA implementation
    df = pd.read_csv(file, sep='\t', lineterminator='\n', header=None)

    x = df.loc[:, 2:].values
    X = x - x.mean(0)
    # x = StandardScaler().fit_transform(x)
    # print(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)

    principalDF = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    groundtruth = pd.DataFrame(data=df.loc[:, 1].values, columns=['Label'])
    finalDf = pd.concat([principalDF, groundtruth], axis=1)

#Hierarchy result
    my_resutl = pd.DataFrame(data = np.array(gen_result), columns = ['Label'])
    my_Df = pd.concat([principalDF, my_resutl ],axis = 1)

    fig = plt.figure(figsize=(16, 8))
    bx = fig.add_subplot(1, 2, 2)
    bx.set_xlabel('Principal Component 1', fontsize=15)
    bx.set_ylabel('Principal Component 2', fontsize=15)
    # bx.set_title('Hierarchical Clustering Result on Cho.txt', fontsize=20)
    bx.set_title('Hierarchical Clustering Result on iyer.txt', fontsize=20)

    targets = [ i for i in range(1,int(k)+1)]
    colors = ['#' +''.join([random.choice('0123456789ABCDEF') for x in range(6)]) for i in range(int(k))]

    for target, color in zip(targets, colors):
        indicesToKeep = my_Df['Label'] == target
        bx.scatter(my_Df.loc[indicesToKeep, 'principal component 1']
                   , my_Df.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    bx.legend(targets)
    bx.grid()
#Ground truth
#####################################
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('Ground Truth', fontsize=20)


    targets = [ i for i in range(1,int(k)+1)]
    colors = ['#' +''.join([random.choice('0123456789ABCDEF') for x in range(6)]) for i in range(int(k))]

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['Label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    # plt.savefig('hierarchy_cho.eps')
    # plt.savefig('hierarchy_iyer.eps')
    plt.show()





if __name__ == "__main__":
    main()