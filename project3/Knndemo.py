import pandas as pd
import numpy as np
import copy


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
    # file = "project3_dataset1.txt"
    # file = "project3_dataset2.txt"
    train = "project3_dataset3_train.txt"
    test = "project3_dataset3_test.txt"
    # file = sys.argv[1]

    k = 2

    data_train = np.array(pd.read_csv(train, sep='\t', lineterminator='\n', header=None).iloc[:, :-1])
    data_test = np.array(pd.read_csv(test, sep='\t', lineterminator='\n', header=None).iloc[:, :-1])
    rownum1 = len(data_train)
    rownum2 = len(data_test)
    colnum = len(data_train[0])
    ground_truth_train = np.array(pd.read_csv(train, sep='\t', lineterminator='\n', header=None).iloc[:, -1])
    ground_truth_test = np.array(pd.read_csv(test, sep='\t', lineterminator='\n', header=None).iloc[:, -1])



    testdata = data_test
    traindata = data_train

    label_test = ground_truth_test
    label_train = ground_truth_train
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
    print(len(classified_test),len(label_test))
    print(classified_test)
    print(label_test)
    acc,pre,recall,fm = accu_cal(label_test, classified_test)


    print(acc, pre, recall, fm)





if __name__ == "__main__":
    main()