import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import sys
from sklearn.cluster import SpectralClustering, KMeans


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

def sim_cal(data, sig):
    length = len(data)
    sim_matrix = np.array([[0]*length for j in range(length)]).astype(np.float64)
    for i in range(length):
        for j in range(i+1, length):
            result = (np.exp(-np.linalg.norm(data[i]-data[j])/(sig**2))).astype(float)
            # result = np.around(result, decimals = 2)
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
    # file = "cho.txt"
    # file = "new_dataset_1.txt"
    file = sys.argv[1]


    data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 2:])

    ground_truth = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 1])

    id = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 0])

    k = int(sys.argv[2])
    # k = 3
# My code should be here. Finally a generated label list called gen_result should be generated
#######################################
    # #
    # for i in range(15,50):
    #     sig = i/10
    #     print(sig)

#matrix generating and normalizing
    # for cho.txt sig = 1
    #rand = 0.8044 jaccard = 0.4263
    #for iyer.txt sig = 3.6
    #rand = 0.8436 jaccard = 0.4172
    sig = float(sys.argv[3])

    simatrix = sim_cal(data, sig)
    dgmatrix = deg_cal(simatrix)
    Lap_mat = dgmatrix - simatrix
    # print(Lap_mat)
    dg_inv = np.linalg.inv(dgmatrix)
    norm_Lap = np.dot(dg_inv,Lap_mat)
    # print(norm_Lap)
# #eigen value and vecgor generation
    eg_value, eg_vector = np.linalg.eig(norm_Lap)

    sorted_indices = np.argsort(eg_value)
#sort eigen values and eigen vectors
    new_index = sorted_indices[0:k]
    new_egvector = eg_vector[:,new_index]
    reduced_dim = new_egvector

# Please input the initial kmeans points in the manner of num1,num2,num3
    array_id = sys.argv[4]
    array_id = array_id.split(',')
    array_id = list(map(int,array_id))
    print(array_id)

    init_points = reduced_dim[array_id]
    km = KMeans(init = init_points, n_clusters = k)

    # km = KMeans(init = 'k-means++', n_clusters = k)
    km.fit(reduced_dim)
    km.labels_
    # print(km.labels_)

########################################

    # gen_result = result.astype(int)
    gen_result = km.labels_ +1


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

#Spectral result
    my_resutl = pd.DataFrame(data = np.array(gen_result), columns = ['Label'])
    my_Df = pd.concat([principalDF, my_resutl ],axis = 1)

    fig = plt.figure(figsize=(16, 8))
    bx = fig.add_subplot(1, 2, 2)
    bx.set_xlabel('Principal Component 1', fontsize=15)
    bx.set_ylabel('Principal Component 2', fontsize=15)
    # bx.set_title('Spectral Clustering Result on cho.txt', fontsize=20)
    bx.set_title('Spectral Clustering Result on cho.txt', fontsize=20)

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
    # plt.savefig('spectral_cho.eps')
    # plt.savefig('spectral_iyer.eps')
    # plt.savefig('hierarchy_cho.eps')
    # plt.savefig('hierarchy_iyer.eps')
    plt.show()



if __name__ == "__main__":
    main()