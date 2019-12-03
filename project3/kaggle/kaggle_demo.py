import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#更新一下每个点的权重
def weight_update(alpha , y, predict, w):
    yh = y+predict
    update_yh = np.array([-1 if item == 1 else 1 for item in yh])
    weight_array = []
    for i in range(len(w)):
        weight_array.append(w[i]*np.exp(-alpha*update_yh[i]))
    weight_array = weight_array/(np.sum(weight_array))

    return weight_array

def predict(test, alpha_list, bestN, X_train,Y_train):

    m = len(bestN)
    predict_result = []
    for item in test:
        result = 0
        for i in range(m):
            model = KNeighborsClassifier(n_neighbors=bestN[i], weights='distance')
            model.fit(X_train, Y_train)
            item = item.reshape(1,-1)
            if int(model.predict(item)[0]) == 0:
                temp = -1
            else:
                temp = 1
            pre_result = temp*alpha_list[i]
            result += pre_result
        if result>0:
            res = 1
        else:
            res = 0
        predict_result.append(res)
    return predict_result

def main():
    file1 = "train_features.csv"
    file2 = "train_label.csv"
    testfile = "test_features.csv"
    test = np.array(pd.read_csv(testfile, sep=',', lineterminator='\n', header=None).iloc[:,1:])
    data = np.array(pd.read_csv(file1, sep=',', lineterminator='\n', header=None).iloc[:,1:])
    # id = np.array(pd.read_csv(file1, sep=',', lineterminator='\n', header=None).iloc[:,0])
    ground_truth = np.array(pd.read_csv(file2, sep=',', lineterminator='\n', header=None).iloc[1:,1])
    X_train, X_test, Y_train, Y_test = train_test_split(data,ground_truth, test_size=0.2, random_state=16)

    m = len(X_test)
    weights = np.array([1 / m for i in range(m)])  # m个弱分类器，初始权重是一样的
    Y_test = np.array(list(map(int, Y_test)))

    alpha_list = []

    T = 4

    for model_num in range(T):
        if model_num == 3:
            logit_re = LogisticRegression()
            logit_re.fit(X_train, Y_train)

            result = logit_re.predict(X_test)  # 当有i个最近邻点当作neighbor的时候，得到的结果
            result = np.array(list(map(int, result)))
            new = result + Y_test
            e_row = np.array([1 if item == 1 else 0 for item in new])
            error = np.sum(e_row * weights)


            alpha = 0.5 * (np.log((1 - error) / error))  # 弱分类器的权重(通过error来算出来)
            alpha_list.append(alpha)  # 记录下弱分类器的权重list
            weights = weight_update(alpha, Y_test, result, weights)  # Y_test：真实值 final：预测

            #score_lr = logit_re.score(X_test, Y_test)

        elif model_num==1:
            svm = SVC()
            svm.fit(X_train, Y_train)

            result = svm.predict(X_test)  # 当有i个最近邻点当作neighbor的时候，得到的结果
            result = np.array(list(map(int, result)))
            new = result + Y_test
            e_row = np.array([1 if item == 1 else 0 for item in new])
            error = np.sum(e_row * weights)

            alpha = 0.5 * (np.log((1 - error) / error))  # 弱分类器的权重(通过error来算出来)
            alpha_list.append(alpha)  # 记录下弱分类器的权重list
            weights = weight_update(alpha, Y_test, result, weights)  # Y_test：真实值 final：预测


            #score_svm = svm.score(X_test, Y_test)

        elif model_num==2:
            dt = DecisionTreeClassifier()
            dt.fit(X_train, Y_train)

            result = dt.predict(X_test)  # 当有i个最近邻点当作neighbor的时候，得到的结果
            result = np.array(list(map(int, result)))
            new = result + Y_test
            e_row = np.array([1 if item == 1 else 0 for item in new])
            error = np.sum(e_row * weights)

            alpha = 0.5 * (np.log((1 - error) / error))  # 弱分类器的权重(通过error来算出来)
            alpha_list.append(alpha)  # 记录下弱分类器的权重list
            weights = weight_update(alpha, Y_test, result, weights)  # Y_test：真实值 final：预测


        elif model_num==0:
            rf = RandomForestClassifier(n_estimators=500,random_state=666, n_jobs=-1)
            rf.fit(X_train, Y_train)

            result = rf.predict(X_test)  # 当有i个最近邻点当作neighbor的时候，得到的结果
            result = np.array(list(map(int, result)))
            new = result + Y_test
            e_row = np.array([1 if item == 1 else 0 for item in new])
            error = np.sum(e_row * weights)

            alpha = 0.5 * (np.log((1 - error) / error))  # 弱分类器的权重(通过error来算出来)
            alpha_list.append(alpha)  # 记录下弱分类器的权重list
            weights = weight_update(alpha, Y_test, result, weights)  # Y_test：真实值 final：预测



    print(alpha_list)
####################################### this is the predicting part #############################################
    predict_result=[]
    for item in test:
        result = 0
        for i in range(T):
            #result = logit_re.predict(X_test)
            #result = logit_re.predict(X_test)

            #result = svm.predict(X_test)

            #result = dt.predict(X_test)

            if i==3:
                item = item.reshape(1, -1)
                if int(logit_re.predict(item)[0]) == 0:
                    temp = -1
                else:
                    temp = 1
                pre_result = temp * alpha_list[i]
                #print("pre_result is"+str(pre_result))
                result += pre_result

            elif i==1:
                item = item.reshape(1, -1)
                if int(svm.predict(item)[0]) == 0:
                    temp = -1
                else:
                    temp = 1
                pre_result = temp * alpha_list[i]
                result += pre_result

            elif i==2:
                item = item.reshape(1, -1)
                if int(dt.predict(item)[0]) == 0:
                    temp = -1
                else:
                    temp = 1
                pre_result = temp * alpha_list[i]
                result += pre_result


            elif i==0:
                item = item.reshape(1, -1)
                if int(rf.predict(item)[0]) == 0:
                    temp = -1
                else:
                    temp = 1
                pre_result = temp * alpha_list[i]
                result += pre_result

        if result > 0:
            res = 1
        else:
            res = 0
        predict_result.append(res)

    # from sklearn.metrics import accuracy_score
    #
    # accu_score = accuracy_score(Y_test, predict_result)
    # print(accu_score)

    res = pd.DataFrame(columns=["id", "label"])
    res["label"] = predict_result
    res["id"] = np.array([418 + i for i in range(len(predict_result))])
    print(res)
    print(predict_result)
    res.to_csv("result.csv", index=False)
















    #print(alpha_list)
    #final_result = predict(test, alpha_list, best_neighbour, X_train, Y_train)



            #score_dt = dt.score(X_test, Y_test)





























if __name__ == "__main__":
    main()