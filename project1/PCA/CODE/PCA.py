import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
################################################
#Preprocess the data, read all data in the list

def preprocess(string): # Three choices pca_a.txt, pca_b.txt, pca_c.txt
    with open(string,'r')as f:
        data = f.readlines()
        data_list = []
        for line in data:
            line = line.strip('\n')
            pca = line.split(sep="\t")
            data_list.append(pca)
        return data_list


##################################################
#Calculate the covariance, eigen vector and eigen value of the data and perform the
#matrix multiplication on the raw data by eigen vector



def transform_data(data,rank):
    data = np.array(data)
    label = data[:,-1]
    num_data = data[:,:-1]
    num_data = np.array(num_data).astype(np.float)
    # mean substraction
    adjusted_data = num_data - num_data.mean(0)


    cov = np.cov(adjusted_data.T)

    eg_value, eg_vector = np.linalg.eig(cov)
    final_data = np.dot(eg_vector[rank].reshape(1, len(eg_vector[rank])), adjusted_data.T)
    return label, final_data.flatten()

#######################################################
#classify the data and plot them in a 2D graph

def classify_and_plot(label, x_axis,y_axis,name):
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
    plt.title(name)
    # plt.savefig(name+'.eps')
    plt.show()


##############################################
#svd decomposition, return the first two major column component and the label
def svd_plot(data):
    data = np.array(data)
    label = data[:,-1]
    num_data = data[:,:-1]
    num_data = np.array(num_data).astype(np.float)
    u, s, vt = np.linalg.svd(num_data.T)
    pivot_vector = u[:,:2]
    svd_plot = np.dot(pivot_vector.reshape(2,len(pivot_vector)),num_data.T)

    return label ,svd_plot

################################################
#t-SNE dimention reduction
def tsne(data):
    data = np.array(data)
    label = data[:, -1]
    num_data = data[:, :-1]
    num_data = np.array(num_data).astype(np.float)

    pca_tsne = manifold.TSNE(n_components=2,init = 'pca')
    pca_tsne.fit_transform(num_data)
    newMat = pca_tsne.embedding_.T

    return label, newMat

def main():
     dataset1 = preprocess('pca_a.txt')
     dataset2 = preprocess('pca_b.txt')
     dataset3 = preprocess('pca_c.txt')
     dataset4 = preprocess('pca_demo.txt')

     label1, transformed_data1_rank_1 = transform_data(dataset1,0)
     label1, transformed_data1_rank_2 = transform_data(dataset1,1)
     label2, transformed_data2_rank_1 = transform_data(dataset2,0)
     label2, transformed_data2_rank_2 = transform_data(dataset2,1)
     label3, transformed_data3_rank_1 = transform_data(dataset3,0)
     label3, transformed_data3_rank_2 = transform_data(dataset3,1)
     label4, transformed_data4_rank_1 = transform_data(dataset4,0)
     label4, transformed_data4_rank_2 = transform_data(dataset4,1)

     classify_and_plot(label1, transformed_data1_rank_2, transformed_data1_rank_1,'PCA_a')
     classify_and_plot(label2, transformed_data2_rank_2, transformed_data2_rank_1,'PCA_b')
     classify_and_plot(label3, transformed_data3_rank_2, transformed_data3_rank_1,'PCA_c')
     classify_and_plot(label4, transformed_data4_rank_2, transformed_data4_rank_1,'PCA_demo')
     lab_svd1, svd_coor1 = svd_plot(dataset1)
     lab_svd2, svd_coor2 = svd_plot(dataset2)
     lab_svd3, svd_coor3 = svd_plot(dataset3)
     lab_svd4, svd_coor4 = svd_plot(dataset4)

     classify_and_plot(lab_svd1, svd_coor1[0], svd_coor1[1],'svd_a')
     classify_and_plot(lab_svd2, svd_coor2[0], svd_coor2[1],'svd_b')
     classify_and_plot(lab_svd3, svd_coor3[0], svd_coor3[1],'svd_c')
     classify_and_plot(lab_svd4, svd_coor4[0], svd_coor4[1],'svd_demo')

     lab_tsne1, tsne_coor1 = tsne(dataset1)
     lab_tsne2, tsne_coor2 = tsne(dataset2)
     lab_tsne3, tsne_coor3 = tsne(dataset3)
     lab_tsne4, tsne_coor4 = tsne(dataset4)

     classify_and_plot(lab_tsne1, tsne_coor1[0], tsne_coor1[1],'tsne_a')
     classify_and_plot(lab_tsne2, tsne_coor2[0], tsne_coor2[1],'tsne_b')
     classify_and_plot(lab_tsne3, tsne_coor3[0], tsne_coor3[1],'tsne_c')
     classify_and_plot(lab_tsne4, tsne_coor4[0], tsne_coor4[1],'tsne_demo')




if __name__ == "__main__":
    main()

