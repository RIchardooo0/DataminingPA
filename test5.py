import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
import itertools


def preprocess(): # Three choices pca_a.txt, pca_b.txt, pca_c.txt
    with open('associationruletestdata.txt','r')as f:
        data = f.readlines()
        data_list = []
        for line in data:
            line = line.strip('\n')
            pca = line.split(sep="\t")
            data_list.append(pca)
        return data_list

"""
def set_generation(itemset, length, int):
    list1 = []
    for i in combinations(itemset, int):
        if len(list(set(i[0]+i[1]))) == length:
            a = list(set(i[0] + i[1]))
            # list1.append(list(set(i[0]+i[1])))
            list1.append(a)

    return list1
"""


def main():
    Data = preprocess()
    Dat = pd.DataFrame(Data)


    for i in range(len(Dat.columns) - 1):
        if i<9:
         Dat[i] = 'G' + "0"+str(i + 1) + "_" + Dat[i].astype(str)
        else:
         Dat[i] = 'G' +  str(i + 1) + "_" + Dat[i].astype(str)
    Data_set = Dat
    support = input("Please input the support\t")
    single_candidate = set()
    ##################有个dict记录出现次数
    dict={}

    for i in range(len(Dat.columns)):
        dat_col = Dat[i].groupby(Dat[i]).describe()
        for j in range(len(dat_col)):
            count = list(dat_col.iloc[j])[2:4]
            if (count[1] >= int(support)):
                single_candidate.add(count[0])
                dict[str(count[0])]=count[1]
    #single_candidate = [[i] for i in single_candidate]
    print('number of length-1 frequent itemsets:\n'+ str(len(single_candidate)))
    data_list = []

    for i in range(len(Data)):
        row = list(Data_set.iloc[i])
        data_list.append(row)
    print(data_list)
    next_level = single_candidate
    print(next_level)
    print(dict)

    next_level_raw = list(itertools.combinations(single_candidate, 2))
    print(next_level_raw)
    next_level = []
    for item_son in next_level_raw:
        count = 0
        for item_father in data_list:
            # count2+=1
            # print(count2)
            # print("!!!!!")
            # print(set(item_son))
            #print(item_father)
            if set(item_son).issubset(set(item_father)):
                count = count + 1
        # print(count)
        if count >= int(support):
            # print("^^^^^^^")
            next_level.append(item_son)
            dict[str(set(item_son))] = count
    print(len(next_level))
    print(next_level)
    print(dict)

    print("number of length-2 frequent itemsets:"+str(len(next_level)))
###########################################################
#这个是从三开始


    for length in range(3, len(Dat.columns)):
        next_level_new = []
        #从小到大排序
        for i in range(len(next_level)):
            list_temp = []
            list_temp = list(next_level[i])
            list_temp.sort()
            next_level_new.append(list_temp)

        print(next_level_new)
        #组合过程，从开始到n-2相同的，组合
        next_level_new1 = []
        for x in range(len(next_level_new)):
            for y in range(x+1,len(next_level_new)):
                L1 = next_level_new[x][:length-2]
                #print("L1 is")
                #print(L1)
                L2 = next_level_new[y][:length-2]
                #print("L2 is")
                 #print(L2)
                if(L1==L2):
                   next_level_new1.append(list(set(next_level_new[x] + next_level_new[y])))
        print(next_level_new1)
        print(len(next_level_new1))

        list_temp1=[]
        geshu = 0 #遍历，外层儿子里层爸爸，开始频繁项集的个数
        for item_son in next_level_new1:
            count = 0
            for item_father in data_list:
               if set(item_son).issubset(set(item_father)):
                  count = count + 1
             # print(count)
            if count >= int(support):
               # print("^^^^^^^")
                dict[str(set(item_son))] = count
                list_temp1.append(item_son)
                geshu = geshu+1
        if count == 0:
            print("No more rules")
            break
        next_level = list_temp1
        print("number of length-"+str(length)+" frequent item sets is "+str(geshu))
    '''
    lis_new = []
    [lis_new.append(i) for i in next_level_new1 if not i in lis_new]
    
    print("&&&&&&&")
    print(len(lis_new))
    '''
'''
    next_level
    for i in range(len(next_level_new1)):
        list_temp = []
        list_temp = list(next_level_new1[i])
        list_temp.sort()
        next_level_new.append(list_temp)
'''













"""
    
        lis = set_generation(next_level,length,2)

        lis_new = []
        [lis_new.append(i) for i in lis if not i in lis_new]
        #print(lis)
        next_level = []
        for i in lis_new:
            counter = 0
            for j in data_list:
                if set(i).issubset(j):
                    counter+=1
            if counter>=int(support):
                next_level.append(i)
        if len(next_level)==0:
            print("No more rules")
            break
        print('number of length-'+str(length)+'\tfrequent itemsets:\t'+ str(len(next_level)))
        print(next_level)
   """





    # for comb in range(len(Data.columns)):
    #     lis = set_generation(single_candidate, comb)
    #     for pair in lis:
    #


if __name__ == "__main__":
    main()