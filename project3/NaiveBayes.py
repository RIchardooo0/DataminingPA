import numpy as np
import pandas as pd
import random

def preprocess(data):
    data_str = data.copy(deep=True)
    data_val = data.copy(deep=True)
    for i in range(data.shape[1]):
        if type(data.iloc[0,i]) == str:
            data_val = data_val.drop([i],axis=1)
        else:
            data_str = data_str.drop([i],axis=1)
    data_str = pd.concat([data_str,data.iloc[:,-1]],axis= 1,ignore_index= True)
    return data_str, data_val
def split_data(data):
    interval = int(data.shape[0]/10)
    acc_list = []
    pre_list = []
    recall_list = []
    fm_list = []
    for i in range(10):
        if i == 9:
            test_data = data.iloc[i*interval:,:]
            train_data = data.iloc[:i*interval,:]
        else:
            test_data = data.iloc[i*interval:(i+1)*interval,:]
            train_data = data.iloc[:i*interval,:]
            train_data = pd.concat([train_data,data.iloc[(i+1)*interval:,:]],axis=0,ignore_index = True)
        predict = naive_bayes(data, test_data, train_data)
        label_test = test_data.iloc[:,-1]
        acc,pre,recall,fm = accu_cal(label_test, predict)
        acc_list.append(acc)
        pre_list.append(pre)
        recall_list.append(recall)
        fm_list.append(fm)
#     print(acc_list, pre_list, recall_list, fm_list)
    tru_acc = np.mean(acc_list)
    tru_pre = np.mean(pre_list)
    tru_recall = np.mean(recall_list)
    tru_fm = np.mean(fm_list)
    print(tru_acc,tru_pre,tru_recall,tru_fm)
def naive_bayes(data, test_data, train_data):
    train_data_str, train_data_val = preprocess(train_data)
    test_data_str, test_data_val = preprocess(test_data)
    if train_data_str.shape[1] == 1:
        labels = data.iloc[:,-1].value_counts().index
        means = []
        variances = []
        res = []
        nums1 = []
        nums2 = []
        for label in labels:
            train_data_label = train_data.loc[train_data.iloc[:,-1] == label,:]
            if (label == 0):
                nums1.append(train_data_label.shape[0])
            elif (label == 1):
                nums2.append(train_data_label.shape[0])
            mean = train_data_label.iloc[:,:-1].mean()
            variance = np.sum((train_data_label.iloc[:,:-1]- mean)**2)/(train_data_label.shape[0])
            means.append(mean)
            variances.append(variance)
        means = pd.DataFrame(means, index = labels)
        variances = pd.DataFrame(variances, index = labels)
        for i in range(test_data.shape[0]):
            temp = test_data.iloc[i,:-1] - means
            probability = 1
            prob = np.exp(-1*(temp)**2/(variances*2))/np.sqrt(2*np.pi*variances)
            for j in range(test_data.shape[1] - 1):
                probability = probability*prob.iloc[:,j]
            probability[0] = probability[0]*(nums1[0]/train_data.shape[0])
            probability[1] = probability[1]*(nums2[0]/train_data.shape[0])
            pre = np.argmax(probability)
            res.append(pre)
    elif train_data_val.shape[1] == 1:
        labels = data.iloc[:,-1].value_counts().index
        res = []
        pros = [None]*2
        for i in range(test_data.shape[0]):
            for label in labels:
                probability = 1
                train_data_label = train_data.loc[train_data.iloc[:,-1] == label,:]
                for j in range(test_data.shape[1] - 1):
                    count = train_data_label.iloc[:,j].value_counts()
                    try:
                        number = count[test_data.iloc[i,j]]
                        probability = probability*(number/train_data_label.shape[0])
                    except:
                        print("zero probability")
                        probability = 1/train_data_label.shape[0]
                probability = probability*(train_data_label.shape[0]/train_data.shape[0])
                if label == 0:
                    pros[0] = probability
                elif label == 1:
                    pros[1] = probability
            print("p(X|H0)p(H0) = " + str(pros[0]))
            print("p(X|H1)p(H1) = " + str(pros[1]))
            pre = np.argmax(pros)
            pros.clear()
            pros = [None]*2
            res.append(pre)
    else:
        labels = data.iloc[:,-1].value_counts().index
        means = []
        variances = []
        res = []
        nums = []
        pro1 = []
        pro2 = []
        pro3 = []
        pro4 = []
        for label in labels:
            train_data_label = train_data_val.loc[train_data_val.iloc[:,-1] == label,:]
            nums.append(train_data_label.shape[0])
            mean = train_data_label.iloc[:,:-1].mean()
            variance = np.sum((train_data_label.iloc[:,:-1]- mean)**2)/(train_data_label.shape[0])
            means.append(mean)
            variances.append(variance)
        means = pd.DataFrame(means, index = labels)
        variances = pd.DataFrame(variances, index = labels)
        for i in range(test_data_val.shape[0]):
            temp = test_data_val.iloc[i,:-1] - means
            probability = 1
            prob = np.exp(-1*(temp)**2/(variances*2))/np.sqrt(2*np.pi*variances)
            for j in range(test_data_val.shape[1] - 1):
                probability = probability*prob.iloc[:,j]
            probability[0] = probability[0]*(nums[0]/train_data.shape[0])
            probability[1] = probability[1]*(nums[1]/train_data.shape[0])
            pro1.append(probability[0])
            pro2.append(probability[1])
        for i in range(test_data_str.shape[0]):
            for label in labels:
                pro = 1
                train_data_label_str = train_data_str.loc[train_data_str.iloc[:,-1] == label,:]
                for j in range(test_data_str.shape[1] - 1):
                    count = train_data_label_str.iloc[:,j].value_counts()
                    try:
                        number = count[test_data_str.iloc[i,j]]
                        pro = pro*(number/train_data_label_str.shape[0])
                    except:
                        print("zero probability")
                        pro = 1/train_data_label_str.shape[0]
                pro = pro*(train_data_label_str.shape[0]/train_data_str.shape[0])
                if label == 0:
                    pro3.append(pro)
                elif label == 1:
                    pro4.append(pro)
        for i in range(len(pro3)):
            if pro1[i]*pro3[i] >= pro2[i]*pro4[i]:
                res.append(0)
            else:
                res.append(1)
    return res
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

data = pd.read_csv('project3_dataset1.txt', sep='\t', header = None)
split_data(data)
