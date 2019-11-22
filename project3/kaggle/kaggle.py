import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


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
            pre_result = np.array(model.predict(item)[0])*alpha_list[i]
            result =+  pre_result

        predict_result.append(result)
    return predict_result

def main():
    file1 = "train_features.csv"
    file2 = "train_label.csv"
    test = np.array(pd.read_csv(file1, sep=',', lineterminator='\n', header=None).iloc[4:9,1:])
    data = np.array(pd.read_csv(file1, sep=',', lineterminator='\n', header=None).iloc[:,1:])
    id = np.array(pd.read_csv(file1, sep=',', lineterminator='\n', header=None).iloc[:,0])
    ground_truth = np.array(pd.read_csv(file2, sep=',', lineterminator='\n', header=None).iloc[1:,1])
    X_train, X_test, Y_train, Y_test = train_test_split(data,ground_truth, test_size=0.2, random_state=22)

    #initialize the weight of training test
    m = len(X_test)
    weights = np.array([1/m for i in range(m)])
    T = 20
    best_neighbour = [] # record the parameter for weak classifier
    alpha_list = []
    #num of iterations T
    for i in range(T):
        #find the best n_neighbors to seperate the data with least error
        errormin = 99999
        for i in range(1,100):
            model1 = KNeighborsClassifier(n_neighbors=i, weights = 'distance')
            model1.fit(X_train, Y_train)
            result = model1.predict(X_test)
            new = result + Y_test
            e_row = np.array([1 if item == 1 else 0 for item in new])
            error = np.sum(e_row*weights)
            if error < errormin:
                errormin = error
                index = i
                final = result
        if errormin> 0.5: break

        alpha = 0.5*(np.exp((1-errormin)/errormin))
        best_neighbour.append(index)
        alpha_list.append(alpha)
        weights = weight_update(alpha, Y_test, final, weights)
    print(best_neighbour,alpha_list)
    final_result = predict(test,alpha_list,best_neighbour, X_train, Y_train)
    print(final_result)



            # print(score1)
    # print(score1, score2, score3)



if __name__ == "__main__":
    main()