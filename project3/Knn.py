import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import sys


def norm_data(matrix):
    # new_matrix = np.zeros(matrix.shape)
    rowlen = len(matrix)
    new_matrix = [[] for i in range(rowlen)]

    for i in range(len(matrix[0])):
        if type(matrix[0][i]) == float or type(matrix[0][i]) == int:
            my_col = matrix[:,i]
            col_max = np.max(my_col)
            col_min = np.min(my_col)
            de = col_max - col_min
            for j in range(rowlen):
                new_matrix[j].append((my_col[j] - col_min)/de)
        else:
            for j in range(rowlen):
                new_matrix[j].append(np.array(matrix[j][i]))

    return np.array(new_matrix)

def accu_cal(truth, result):
    total = len(truth)
    tp = np.sum(truth*result)
    plus = truth - result
    fp = sum([1 for i in plus if i == -1])
    fn = sum([1 for i in plus if i == 1])
    tn = total - tp - fp - fn

    # print(tp, fp, fn, tn)
    acc = (tp+tn)/total

    if tp + fp != 0:
        pre = tp / (tp + fp)
    else:
        pre = 0
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if 2 * tp + fn + fp != 0:
        fm = 2 * tp / (2 * tp + fn + fp)
    else:
        fm = 0
    return acc,pre,recall,fm




def main():
    file = "project3_dataset1.txt"
    # file = "project3_dataset2.txt"
    # file = "project3_dataset4.txt"
    # file = sys.argv[1]

    k = 2

    data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, :-1])
    rownum = len(data)
    colnum = len(data[0])
    ground_truth = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, -1])
    ground_truth = ground_truth.reshape(rownum,1)
    copy_data = data
    flag = 0
    str_attr = []
    category = []
    for i in range(colnum):
        if type(data[0][i]) == str:
            flag = 1
            str_attr.append(i)
            category1 = pd.Categorical(data[:,i]).categories
            print(category1)
            category.append(category1)
    if flag == 1:
        copy_data = copy.deepcopy(data)
        for num in range(len(category)):
            cat_num = len(category[num])
            cat_mapping = {}
            index = [0 for i in range(cat_num-1)]
            index.append(1/np.sqrt(cat_num))
            for i in category[num]:
                cat_mapping[i] = index
                index = copy.deepcopy(index)
                index.pop(0)
                index.append(0)
            new = pd.Series(data[:,str_attr[num]]).map(cat_mapping).values
            for j in range(len(copy_data)):
                copy_data[j][str_attr[num]] = np.array(new[j])
    checklen = round(rownum/10)
    normed_data = np.hstack((norm_data(copy_data),ground_truth))

    acc_list = []
    pre_list = []
    recall_list = []
    fm_list = []

    for i in range(10):

        if i == 9:
            testdata = normed_data[i*checklen:,:]
            traindata = normed_data[:i*checklen,:]
        else:
            testdata = normed_data[i*checklen:(i+1)*checklen,:]
            traindata = normed_data[:i*checklen,:]
            traindata = np.vstack((traindata,normed_data[(i+1)*checklen:,:]))
        label_test = testdata[:,-1]
        label_train = traindata[:,-1]
        classified_test = []
        for item in testdata:
            unordered = []
            for train in traindata:
                z = item[:-1] - train[:-1]
                z_modified = [ i if type(i) == int or type(i) == float else np.linalg.norm(i) for i in z  ]
                x_norm = np.linalg.norm(z_modified)
                unordered.append(x_norm)
            order = np.argsort(unordered)
            order = order[:k]
            classifiedlabel = [label_train[i] for i in order]
            if sum(classifiedlabel) >= k/2:
                classified_test.append(1)
            else:
                classified_test.append(0)

        acc,pre,recall,fm = accu_cal(label_test, classified_test)
        acc_list.append(acc)
        pre_list.append(pre)
        recall_list.append(recall)
        fm_list.append(fm)

    print(acc_list, pre_list, recall_list, fm_list)

    tru_acc = np.mean(acc_list)
    tru_pre = np.mean(pre_list)
    tru_recall = np.mean(recall_list)
    tru_fm = np.mean(fm_list)

    print(tru_acc,tru_pre,tru_recall,tru_fm)





if __name__ == "__main__":
    main()