import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import sys


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

def sim_cal(data, eps, sig):
    length = len(data)
    sim_matrix = np.array([[0]*length for j in range(length)]).astype(np.float64)
    for i in range(length):
        for j in range(i+1, length):
            result = np.exp(-np.linalg.norm(data[i]-data[j])/(sig**2))
            # result = np.around(result, decimals = 2)
            if result > eps:
                sim_matrix[i][j] = result
                sim_matrix[j][i] = result
    return sim_matrix

def deg_cal(matrix):
    result_mat = np.array([[0]*len(matrix) for i in range(len(matrix))]).astype(np.float64)
    sum_res = np.sum(matrix , axis = 1)
    for i in range(len(matrix)):
        result_mat[i][i] = sum_res[i]

    return result_mat




def main():
    # file = "iyer.txt"
    file = "cho.txt"
    # file = sys.argv[1]

    data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 2:])

    ground_truth = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 1])

    id = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 0])

    # k = sys.argv[2]
    k = 5
# My code should be here. Finally a generated label list called gen_result should be generated
#######################################
    eps = 0.5
    sig = 4
#matrix generating and normalizing
    simatrix = sim_cal(data, eps, sig)

    dgmatrix = deg_cal(simatrix)
    Lap_mat = dgmatrix - simatrix
    dg_inv = np.linalg.inv(dgmatrix)
    norm_Lap = np.dot(dg_inv,Lap_mat)
    # print(norm_Lap)
# #eigen value and vecgor generation
    cov = np.cov(norm_Lap)
    eg_value, eg_vector = np.linalg.eig(cov)
    # print(eg_value)
    # print(eg_vector)
    sorted_indices = eg_value.argsort()
#sort eigen values and eigen vectors
    new_egvalue = eg_value[sorted_indices]
    new_egvector = eg_vector[sorted_indices]

    lambda1 = new_egvalue[:-1]
    lambda2 = new_egvalue[1:]
    my_k_list = lambda2 - lambda1
    # print(my_k_list)
    my_k = my_k_list.argmax()+2
    reduced_dim = eg_vector[:my_k]
    print(reduced_dim)
    # print(my_k)







########################################
'''
    gen_result = [0 for i in range(len(ground_truth))]


    inci_truth = incidence_mat_gen(ground_truth)
    inci_hier = incidence_mat_gen(gen_result)

    rand, jaccard = ja_rand_cal(inci_truth, inci_hier)
    print(rand, jaccard)

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
    bx.set_title('Ground Truth', fontsize=20)

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
    ax.set_title('My Clustering Result', fontsize=20)


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

'''



if __name__ == "__main__":
    main()