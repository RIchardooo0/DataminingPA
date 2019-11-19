import pandas
import numpy

import pandas as pd
import numpy as np
import random

data = np.loadtxt('project3_dataset1.txt',delimiter='\t')

print(data)

def calculateGiNi(data):
    last_col = data.shape[1]
    total_num = data
    dict = {}  #1有几个，0有几个
    result = 1

    for item in data:
        if data[last_col] in dict:
            dict[item[last_col-1]] = dict[item[last_col-1]]+1
        else:
            dict[item[last_col-1]] = 1

    for dict_item in dict:
        gigi_sub = dict[dict_item]/total_num
        result = result - gigi_sub*gigi_sub

    return result

def decide_split_node(data):
    col_num = data.shape[1]
    line_num = data.shape[0]
    best_gini = 1000

    for col in range(0,col_num):
        for line_num in range(0,line_num):
            left = []
            right = []
            count_left = 0
            count_right = 0
            for line in data:
                if line[col]<data[line_num][col]:
                    left.append(line)
                else:
                    right.append(line)
            gini_left = calculateGiNi(left)
            gini_right = calculateGiNi(right)

            pa_left = left.shape[1]/line_num
            pa_right = right.shape[1]/line_num

            gini = pa_left*gini_left+pa_right*gini_right

        if best_gini>gini:
            best_gini = gini
            data_after_split = left,right
            value = line[col] #this is the value it should split first
            final_col = col

    return {'best_gini': best_gini, 'value': value, 'left_right_data': data_after_split, 'col': final_col}


def splitNode(node):
    left,right = node['data_after_split']
    len_left = len(left)
    len_right = len(right)
    dict_left = {}
    dict_right = {}

    if node['best_gini']==0: #如果GINI是零的话，就决出了结果
        if len_left>=1:
            last_col_left = left.shape[1]
            for item in left:
                if left[last_col_left - 1] in dict_left:
                    dict_left[item[last_col_left - 1]] = dict_left[item[last_col_left - 1]] + 1
                else:
                    dict_left[item[last_col_left - 1]] = 1

            node['left'] = max(dict_left)

        if len_right>=1:
            last_col_right = right.shape[1]
            for item in right:
                if right[last_col_right - 1] in dict_right:
                    dict_right[item[last_col_right - 1]] = dict_right[item[last_col_right - 1]] + 1
                else:
                    dict_right[item[last_col_right - 1]] = 1
            node['left'] = max(dict_left)

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













    if row[node['index']] < node['val']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']




def create_tree(data):
    root = decide_split_node(data)
    splitNode(root)
















    #创建完成dist



