import numpy as np
import pandas as pd
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

def classify_and_plot(label, x_axis,y_axis,num):
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
    plt.subplot(1,3,num)
    for i in data_group:
        group_x = []
        group_y = []
        for j in i:
            group_y.append(y_axis[j])
            group_x.append(x_axis[j])
        plt.scatter(group_x, group_y, c=color[index3], marker='.')
        index3 += 1


def svd_plot(data):
    u, s, vt = np.linalg.svd(data)


def main():
    dataset1 = preprocess('pca_a.txt')
    dataset2 = preprocess('pca_b.txt')
    dataset3 = preprocess('pca_c.txt')


    label1, transformed_data1_rank_1 = transform_data(dataset1,0)
    label1, transformed_data1_rank_2 = transform_data(dataset1,1)
    label2, transformed_data2_rank_1 = transform_data(dataset2,0)
    label2, transformed_data2_rank_2 = transform_data(dataset2,1)
    label3, transformed_data3_rank_1 = transform_data(dataset3,0)
    label3, transformed_data3_rank_2 = transform_data(dataset3,1)

    classify_and_plot(label1, transformed_data1_rank_2, transformed_data1_rank_1,1)
    classify_and_plot(label2, transformed_data2_rank_2, transformed_data2_rank_1,2)
    classify_and_plot(label3, transformed_data3_rank_2, transformed_data3_rank_1,3)
    plt.show()



    # data_directory = str('../' + sys.argv[1])
    # data_path = os.listdir(data_directory)
    # img_path = []
    # for imgname in data_path:
    #
    #     img_path.append(imgname)
    # img_path.sort()
    # print(img_path)


if __name__ == "__main__":
    main()
#
# fig = plt.figure(figsize=[12,6])
# plt.subplot(1, 2, 1)
# plt.plot(range(pmax),mses5_train)
# plt.title('MSE for Train Data')
# plt.legend(('No Regularization','Regularization'))
# plt.subplot(1, 2, 2)
# plt.plot(range(pmax),mses5)
# plt.title('MSE for Test Data')
# plt.legend(('No Regularization','Regularization'))
# plt.show()
