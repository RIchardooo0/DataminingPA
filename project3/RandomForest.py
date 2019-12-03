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

def decide_split_node(data,col_dict):
    data = np.array(data)
    print("data is")
    print(data)
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

    ori_final_col = col_dict[final_col]

    print("best_gini is "+str(best_gini)+" col is "+str(final_col)+" ori_col is "+str(ori_final_col)+" value is "+str(value))
    return {'best_gini': best_gini, 'value': value, 'left_right_data': data_after_split, 'col': final_col,'ori_col': ori_final_col}


def splitNode(node,col_dict):
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
            node['right'] = max(dict_left)

        if len_right>=1:

            last_col_right = right.shape[1]
            for item in right:
                if item[last_col_right - 1] in dict_right:
                    dict_right[item[last_col_right - 1]] = dict_right[item[last_col_right - 1]] + 1
                else:
                    dict_right[item[last_col_right - 1]] = 1
            node['left'] = max(dict_right)
            node['right'] = max(dict_right)


    elif node['best_gini']!=0 and (len_left==0 or len_right==0):
        if len_left==0:
            for i in range(0, len_right):
                dict = {}
                if right[i][-1] not in dict:
                    dict[right[i][-1]] = 1
                else:
                    dict[right[i][-1]] = dict[right[i][-1]] + 1
            print("####")
            print(dict)
            print(max(dict, key=dict.get))
            node['right']=max(dict, key=dict.get)
            node['left']=max(dict, key=dict.get)

        elif len_right==0:
            for i in range(0, len_left):
                dict = {}
                if left[i][-1] not in dict:
                    dict[left[i][-1]] = 1
                else:
                    dict[left[i][-1]] = dict[left[i][-1]] + 1
            print("####")
            print(dict)
            print(max(dict, key=dict.get))
            node['right'] = max(dict, key=dict.get)
            node['left'] = max(dict, key=dict.get)

    elif node['best_gini']!=0 and len_left!=0 and len_right!=0:

        node['left'] = decide_split_node(left,col_dict)
        splitNode(node['left'],col_dict)
        node['right'] = decide_split_node(right,col_dict)
        splitNode(node['right'],col_dict)


    # print("left is")
    # print(node['left'])
    # print("right is")
    # print(node['right'])


def predict(node,row):






    if row[node['ori_col']]>=node['value']:
        if isinstance(node['right'], dict):
            #print("it is dict")
            predict(node['right'], row)
        else:
            return node['right']

    elif row[node['ori_col']]<node['value']:
        if isinstance(node['left'], dict):
            #print("it is dict")
            predict(node['left'], row)
        else:
            return node['left']


    if row[node['ori_col']] < node['value']:
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

data_test = np.loadtxt('project3_dataset3_test.txt', delimiter='\t')


#####################################处理数据#######################################
raw_data_line = data.shape[0] #原始数据的行数
raw_data_col = data.shape[1] #原始数据的列数

data_test_line = data_test.shape[0]

#num_feature = int(0.2*(len(data[0])-1))
num_feature = 2
num_sample = 20
num_tree = 5


results = []
for i in range(0,num_tree):

    sample_num_list = [] #这个是取多少多少行
    sample_col_list = [] #取很多列

    for i in range(0,num_feature):
        random_col = np.random.choice(raw_data_col-1)
        sample_col_list.append(random_col)
    print("sample_col_list:")
    print(sample_col_list)


    for i in range(0,num_sample):
        random_index = np.random.choice(raw_data_line)
        sample_num_list.append(random_index)
    print("sample_num_list:")
    print(sample_num_list)

    random_training_set = [] #这个是新的数据，但是没有处理过列
    for i in sample_num_list:
        random_training_set.append(data[i].tolist())

    print("random_training_set:")
    print(random_training_set)

    final_training_set = []
    col_dict = {} #记录下
    for item in random_training_set:
        new_row = []
        kk = 0
        for i in sample_col_list:
            new_row.append(item[i])
            col_dict[kk] = i
            kk = kk + 1
        new_row.append(item[-1])
        final_training_set.append(new_row)


    print(final_training_set)
    print(type(final_training_set))
    root = decide_split_node(final_training_set,col_dict)
    splitNode(root,col_dict)


    result = []
    for row in data_test:
        k = predict(root, row)
        result.append(k)

    results.append(result)
    print(results)

final_result_after_weighting = []
for i in range(0,len(results[0])):
    dict = {}
    for item in results:
        if item[i] not in dict:
            dict[item[i]] = 1
        else:
            dict[item[i]] = dict[item[i]]+1
    final_result_after_weighting.append(max(dict, key=dict.get))

print(final_result_after_weighting)

#random_index= list(np.random.choice(len(traingset), samplenumbers))


length_test = data_test.shape[0]


