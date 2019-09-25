import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
import itertools
import copy


# def set_generation(itemset, length, int):
#     list1 = []
#     for i in combinations(itemset, int):
#         if len(list(set(i[0]+i[1]))) == length:
#             list1.append(list(set(i[0]+i[1])))
#     return list1

def preprocess(): # Three choices pca_a.txt, pca_b.txt, pca_c.txt
    with open('associationruletestdata.txt','r')as f:
        data = f.readlines()
        data_list = []
        for line in data:
            line = line.strip('\n')
            pca = line.split(sep="\t")
            data_list.append(pca)
        return data_list


#def sortthelist()
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
    confidence = input("Please input the confidence\t")
    single_candidate = set()
    ##################有个dict记录出现次数
    dict={}
    ##################有个list记录所有的频繁项集
    all_frequentSet = []

    for i in range(len(Dat.columns)):
        dat_col = Dat[i].groupby(Dat[i]).describe()
        for j in range(len(dat_col)):
            count = list(dat_col.iloc[j])[2:4]
            if (count[1] >= int(support)):
                single_candidate.add(count[0])
                dict[str([count[0]])]=count[1]
                all_frequentSet.append([count[0]])
                #print("################")
                #print(all_frequentSet)



    #single_candidate = [[i] for i in single_candidate]
    print('number of length-1 frequent itemsets:\n'+ str(len(single_candidate)))
    data_list = []

    for i in range(len(Data)):
        row = list(Data_set.iloc[i])
        data_list.append(row)
    # print(data_list)
    # next_level = single_candidate
    # print(next_level)
    # print(dict)

    next_level_raw = list(itertools.combinations(single_candidate, 2))

    # print(next_level_raw)
    next_level = []
    for item_son in next_level_raw:
        item_son_list = list(item_son)
        item_son_list.sort()
        count = 0
        for item_father in data_list:
            if set(item_son_list).issubset(set(item_father)):
                count = count + 1
        if count >= int(support):
            next_level.append(item_son)
            dict[str(item_son_list)] = count
            all_frequentSet.append(item_son_list)
    # print(len(next_level))
    # print(next_level)
    # print(dict)

    print("number of length-2 frequent itemsets:\n"+str(len(next_level)))
###########################################################
#这个是从三开始


    for length in range(3, len(Dat.columns)):
        next_level_new = []
        #从小到大排序
        for i in range(len(next_level)):
            list_temp = list(next_level[i])
            list_temp.sort()
            next_level_new.append(list_temp)
        #排序部分
        # print("after sorting")
        # print(next_level_new)
        #组合过程，从开始到n-2相同的，组合
        next_level_new1 = []

        list_after_sort = []
        for x in range(len(next_level_new)):
            for y in range(x+1,len(next_level_new)):
                L1 = next_level_new[x][:length-2]

                L2 = next_level_new[y][:length-2]

                if(L1==L2):
                   list_after_sort = next_level_new[x]+next_level_new[y]

                   #########
                   list_after_sort= list(set(list_after_sort))#去重
                   #########
                   list_after_sort.sort()
                   next_level_new1.append(list(set(list_after_sort)))
        # print(next_level_new1)
        # print(len(next_level_new1))

        list_temp1=[]
        geshu = 0 #遍历，外层儿子里层爸爸，开始频繁项集的个数
        for item_son in next_level_new1:
            item_son.sort()
            count = 0
            for item_father in data_list:
               if set(item_son).issubset(set(item_father)):
                  count = count + 1
            if count >= int(support):
                dict[str(item_son)] = count
                all_frequentSet.append(item_son)
                list_temp1.append(item_son)
                geshu = geshu+1
        if geshu == 0:
            print("No more rules")
            break
        next_level = list_temp1
        print("number of length-"+str(length)+" frequent item sets is\n"+str(geshu))

    # print(len(dict))
    # print(dict)
    # print(len(all_frequentSet))
    # print(all_frequentSet)


        #dict是每个项集对应的出现次数
        #all_frequentSet是所有的频繁项集
    Chart = pd.DataFrame(columns = ['RULE','BODY','HEAD','CONFIDENCE'])

    counter = 0

    for h in range(len(all_frequentSet)):
        print("oooooooooooo"+str(h))
        if len(all_frequentSet[h])==1: #长度为1的频繁项集不配有规则
            continue
        else:
            length = len(all_frequentSet[h])
            previous = []
            for i in range(length):
                if i==1: #如果候选项集个数等于1

                    for item in all_frequentSet[h]:


                        Set_temp =copy.deepcopy(all_frequentSet[h])
                        retA = [ samething for samething in all_frequentSet[h] if samething in [item]]
                        for divider in retA:
                            Set_temp.remove(divider)

                        Set_temp.sort()

                        conf = dict[str(all_frequentSet[h])]/dict[str(Set_temp)]
                        if conf>=int(confidence)/100:
                            print(str(Set_temp)+"---->"+str([item])+"\tconf is"+str(conf))
                            counter = counter+1
                            previous.append([item])
                            Chart.loc[len(Chart)] = pd.Series({'RULE':str(Set_temp+[item]),'BODY':str(Set_temp),'HEAD':str([item]),'CONFIDENCE':conf})
                        #print(str(conf))
                else:  ##如果候选项集个数超过了1
                    print("############"+str(i))
                    i_length_set = []
                    for x in range(len(previous)):
                        for y in range(x + 1, len(previous)):
                            L1 = previous[x][:i - 2]
                            #print("L1 is")
                            #print(L1)

                            L2 = previous[y][:i - 2]
                            #print("L2 is")
                            #print(L2)
                            if L1 == L2:
                                #print("previous X is" + str(previous[x]))
                                #print("previous Y is" + str(previous[y]))
                                list_after_sort = previous[x] + previous[y]
                                list_after_sort = list(set(list_after_sort))
                                list_after_sort.sort()
                                #i_length_set.append(list_after_sort)
#######################################################################################
                                retA=[]
                                Set_temp = copy.deepcopy(all_frequentSet[h])
                                retA = [samething for samething in all_frequentSet[h] if samething in list_after_sort]
                                for divider in retA:
                                    Set_temp.remove(divider)
                                Set_temp.sort()

                                print("retA is"+str(retA))
                                # divide = list(set(all_frequentSet[h])-set([item])).sort()
                                #print("%%%%%%%%%%$$$$$$$")
                                #print(list_after_sort)
                                #print(all_frequentSet[h])
                                #print(Set_temp)
                                #print("%%%%%%%%%%$$$$$$$")

                                conf = dict[str(all_frequentSet[h])]/dict[str(Set_temp)]
                                #print(conf)

                                if conf >= int(confidence)/100:
                                    print(str(Set_temp) + "---->" + str(list_after_sort)+"conf is"+str(conf))
                                    counter = counter + 1
                                    i_length_set.append(list_after_sort)
                                    Chart.loc[len(Chart)] = pd.Series({'RULE': str(Set_temp+list_after_sort), 'BODY': str(Set_temp), 'HEAD': str(list_after_sort),'CONFIDENCE': conf})
                    print("num_"+str(i)+"is "+str(i_length_set))

                    previous = i_length_set
    print(counter)
    Chart.to_csv('Chart.csv',sep = ',')




    # Data = preprocess()
    # Dat = pd.DataFrame(Data)
    #
    #
    # for i in range(len(Dat.columns) - 1):
    #     Dat[i] = 'G' + str(i + 1) + "_" + Dat[i].astype(str)
    # Data_set = Dat
    # support = input("Please input the support\t")
    # single_candidate = set()
    #
    #
    # for i in range(len(Dat.columns)):
    #     dat_col = Dat[i].groupby(Dat[i]).describe()
    #     for j in range(len(dat_col)):
    #         count = list(dat_col.iloc[j])[2:4]
    #         if (count[1] >= int(support)):
    #             single_candidate.add(count[0])
    # single_candidate = [[i] for i in single_candidate]
    # print('number of length-1 frequent itemsets:\n'+ str(len(single_candidate)))
    # data_list = []
    #
    # for i in range(len(Data)):
    #     data_list.append(Data_set.iloc[i])
    # next_level = single_candidate
    #
    # for length in range(2,len(Dat.columns)):
    #     lis = set_generation(next_level,length,2)
    #     lis_new = []
    #     [lis_new.append(i) for i in lis if not i in lis_new]
    #     print(lis_new)
    #     next_level = []
    #     for i in lis_new:
    #         counter = 0
    #         for j in data_list:
    #             if set(i).issubset(j):
    #                 counter+=1
    #         if counter>=int(support):
    #             next_level.append(i)
    #     if len(next_level)==0:
    #         print("No more rules")
    #         break
    #     print('number of length-'+str(length)+'\tfrequent itemsets:\t'+ str(len(next_level)))
    #


if __name__ == "__main__":
    main()