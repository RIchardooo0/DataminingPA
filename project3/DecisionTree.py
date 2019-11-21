import pandas
import numpy

import pandas as pd
import numpy as np
import random



def calculateGiNi(data):

    if len(data)==0:
        return 0

    data = np.array(data)
    last_col = data.shape[1]
    total_num = data.shape[0]
    dict = {}  #1有几个，0有几个
    result = 1

    for item in data:
        if item[last_col-1] in dict:
            dict[item[last_col-1]] = dict[item[last_col-1]]+1
        else:
            dict[item[last_col-1]] = 1
    #print("dict is"+str(dict))
    for dict_item in dict:
        gigi_sub = dict[dict_item]/total_num
        result = result - gigi_sub*gigi_sub
    #print("result is "+str(result))
    return result

def decide_split_node(data):
    data = np.array(data)
    col_num = data.shape[1]
    line_num = data.shape[0]
    best_gini = 1000

    for col in range(0,col_num-1):
        for row in range(0,line_num):
            left = []
            right = []
            count_left = 0
            count_right = 0
            for line in data:
                if line[col]<data[row][col]:
                    left.append(line)
                else:
                    right.append(line)
            #print("$$$"+str(len(left)))
            gini_left = calculateGiNi(left)
            #print("$$$" + str(len(right)))
            gini_right = calculateGiNi(right)

            pa_left = len(left)/line_num
            pa_right = len(right)/line_num
            #print("pa_left is "+str(pa_left))
            #print("gini_left is " + str(gini_left))
            #print("pa_right is " + str(pa_right))
            #print("gini_right is " + str(gini_right))

            gini = pa_left*gini_left+pa_right*gini_right

            #print("gini is " + str(gini) + " col is " + str(col) + " value is " + str(data[row][col]))

            if best_gini>gini:
                best_gini = gini
                data_after_split = left,right
                value = data[row][col] #this is the value it should split first
                final_col = col
                #print("***")

    print("best_gini is "+str(best_gini)+" col is "+str(final_col)+" value is "+str(value))
    return {'best_gini': best_gini, 'value': value, 'left_right_data': data_after_split, 'col': final_col}


def splitNode(node):
    left,right = node['left_right_data']
    len_left = len(left)
    len_right = len(right)
    right = np.array(right)
    left = np.array(left)

    dict_left = {}
    dict_right = {}

    if node['best_gini']==0: #如果GINI是零的话，就决出了结果
        if len_left>=1:
            last_col_left = left.shape[1]
            for item in left:
                if item[last_col_left - 1] in dict_left:
                    dict_left[item[last_col_left - 1]] = dict_left[item[last_col_left - 1]] + 1
                else:
                    dict_left[item[last_col_left - 1]] = 1

            node['left'] = max(dict_left)

        if len_right>=1:
            last_col_right = right.shape[1]
            for item in right:
                if item[last_col_right - 1] in dict_right:
                    dict_right[item[last_col_right - 1]] = dict_right[item[last_col_right - 1]] + 1
                else:
                    dict_right[item[last_col_right - 1]] = 1
            node['right'] = max(dict_right)

    elif node['best_gini']!=0:

        node['left'] = decide_split_node(left)
        splitNode(node['left'])
        node['right'] = decide_split_node(right)
        splitNode(node['right'])


def predict(node,row):

    if row[node['col']]>=node['value']:
        if isinstance(node['right'], dict):
            predict(node['right'], row)
        else:
            return node['right']

    elif row[node['col']]<node['value']:
        if isinstance(node['left'], dict):
            predict(node['left'], row)
        else:
            return node['left']


    if row[node['col']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

data = np.loadtxt('project3_dataset3_train.txt', delimiter='\t')
print(data)
print(type(data))
root = decide_split_node(data)
splitNode(root)

data_test = np.loadtxt('project3_dataset3_test.txt', delimiter='\t')
length_test = data_test.shape[0]
for row in data_test:
    k = predict(root,row)
    print(k)














    #创建完成dist



