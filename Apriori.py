
import pandas as pd
import itertools
import copy




def preprocess():
    with open('associationruletestdata.txt','r')as f:
        data = f.readlines()
        data_list = []
        for line in data:
            line = line.strip('\n')
            pca = line.split(sep="\t")
            data_list.append(pca)
        return data_list


def queryOption(option, df):
    template1 = "asso_rule.template1"
    template2 = "asso_rule.template2"
    template3 = "asso_rule.template3"
    print(option)
    if (option[:len(template1)] == template1):
        result, cnt ,data= queryTemp1(option[len(template1) + 1: -1], df)
        li = [data[i] for i in result]

        print(li,len(li))
    elif (option[:len(template2)] == template2):
        result, cnt, data = queryTemp2(option[len(template2) + 1: -1], df)
        li = [data[i] for i in result]

        print(li,len(li))
    elif (option[:len(template3)] == template3):
        result ,cnt = queryTemp3(option[len(template3) + 1: -1], df)
        print(result)
        print(len(result))
        # li = [df.iloc[i,:]for i in result]
        # print()
    return


def queryTemp1(template1, df):
    parts = eval(template1)
    total_set = set()
    if parts[0] == "RULE":
        if parts[1] == "ANY":
            for son in parts[2]:
                add_set = set()
                for line in range(len(df.RULE)):
                    print(str(line) + "\n")
                    print(type(df.RULE[line]))

                    father_candidate = df.RULE[line].split(',')
                    print("........." + str(set(father_candidate)))
                    # print(",,,,,,,,,"+str(set(son_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        print("!!!!!!!!!!!\n")
                        add_set.add(line)
                total_set = total_set.union(add_set)
            print(len(total_set), "rows selected")
        elif (parts[1] == "NONE"):
            total_set = set()
            for i in range(len(df.RULE)):
                total_set.add(i)
            for son in parts[2]:
                minus_set = set()

                for line in range(len(df.RULE)):
                    print(str(line) + "\n")
                    print(type(df.RULE[line]))

                    father_candidate = df.RULE[line].split(',')
                    print("........." + str(set(father_candidate)))
                    # print(",,,,,,,,,"+str(set(son_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        minus_set.add(line)
                total_set = total_set.difference(minus_set)
        elif (parts[1] == 1):
            counter = []
            counter_set = set()
            for i in range(len(df.RULE)):
                counter.append(0)
            for son in parts[2]:
                # print(son)
                for line in range(len(df.RULE)):
                    father_candidate = []
                    father_candidate = df.RULE[line].split(',')
                    if set([son]).issubset(set(father_candidate)):
                        counter[line] += 1
            for index in range(len(counter)):
                if counter[index] == 1:
                    counter_set.add(index)
            total_set = counter_set
        data = df.RULE
    elif parts[0] == "HEAD":
        if parts[1] == "ANY":
            for son in parts[2]:
                add_set = set()
                for line in range(len(df.HEAD)):
                    print(str(line) + "\n")
                    print(type(df.HEAD[line]))
                    father_candidate = df.HEAD[line].split(',')
                    print("........." + str(set(father_candidate)))
                    # print(",,,,,,,,,"+str(set(son_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        print("!!!!!!!!!!!\n")
                        add_set.add(line)
                total_set = total_set.union(add_set)
        elif parts[1] == "NONE":
            total_set = set()
            for i in range(len(df.HEAD)):
                total_set.add(i)
            for son in parts[2]:
                minus_set = set()
                for line in range(len(df.HEAD)):
                    print(str(line) + "\n")
                    print(type(df.HEAD[line]))
                    father_candidate = df.HEAD[line].split(',')
                    print("........." + str(set(father_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        print("!!!!!!!!!!!\n")
                        minus_set.add(line)
                total_set = total_set.difference(minus_set)
        elif parts[1] == 1:
            counter = []
            counter_set = set()
            for i in range(len(df.HEAD)):
                counter.append(0)
            for son in parts[2]:
                # print(son)
                for line in range(len(df.HEAD)):
                    father_candidate = []
                    father_candidate = df.HEAD[line].split(',')
                    if set([son]).issubset(set(father_candidate)):
                        counter[line] += 1
            for index in range(len(counter)):
                if counter[index] == 1:
                    counter_set.add(index)
            print(counter)
            total_set = counter_set
            print(len(counter_set))
        data = df.HEAD
    elif parts[0] == "BODY":
        if parts[1] == "ANY":
            for son in parts[2]:
                add_set = set()
                for line in range(len(df.BODY)):
                    print(str(line) + "\n")
                    print(type(df.BODY[line]))
                    father_candidate = df.BODY[line].split(',')
                    print("........." + str(set(father_candidate)))
                    # print(",,,,,,,,,"+str(set(son_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        print("!!!!!!!!!!!\n")
                        add_set.add(line)
            total_set = total_set.union(add_set)
        elif parts[1] == "NONE":
            total_set = set()
            for i in range(len(df.BODY)):
                total_set.add(i)
            for son in parts[2]:
                minus_set = set()
                for line in range(len(df.BODY)):
                    print(str(line) + "\n")
                    print(type(df.RULE[line]))
                    father_candidate = []
                    father_candidate = df.BODY[line].split(',')
                    print("........." + str(set(father_candidate)))
                    if set([son]).issubset(set(father_candidate)):
                        print("!!!!!!!!!!!\n")
                        minus_set.add(line)
            total_set = total_set.difference(minus_set)
        elif parts[1] == 1:
            counter = []
            counter_set = set()
            for i in range(len(df.BODY)):
                counter.append(0)
            for son in parts[2]:
                # print(son)
                for line in range(len(df.BODY)):
                    father_candidate = []
                    father_candidate = df.BODY[line].split(',')
                    if set([son]).issubset(set(father_candidate)):
                        counter[line] += 1
            for index in range(len(counter)):
                if counter[index] == 1:
                    counter_set.add(index)


            total_set = counter_set
        data = df.BODY

    return total_set, len(total_set),data


def queryTemp2(template2, df):
    parts = eval(template2)
    count = 0
    total_set = set()
    if parts[0] == "RULE":
        for line in range(len(df.RULE)):
            father_candidate = []
            father_candidate = df.RULE[line].split(',')
            if len(father_candidate) >= parts[1]:
                count += 1
                total_set.add(line)
        print(count)
        data = df.RULE
    elif parts[0] == "HEAD":
        for line in range(len(df.HEAD)):
            father_candidate = []
            father_candidate = df.HEAD[line].split(',')
            if len(father_candidate) >= parts[1]:
                count += 1
                total_set.add(line)
        print(count)
        data = df.HEAD
    elif parts[0] == "BODY":
        for line in range(len(df.BODY)):
            father_candidate = []
            father_candidate = df.BODY[line].split(',')
            if len(father_candidate) >= parts[1]:
                count += 1
                total_set.add(line)
        data = df.BODY
    return total_set,len(total_set),data


def queryTemp3(template3, df):
    parts = eval(template3)
    if parts[0] == "1or1":
        count_set = set()
        count1_set, count1, data1 = queryTemp1(str(parts[1:4]), df)
        count2_set, count2, data2 = queryTemp1(str(parts[4:]), df)
        count_set = count1_set.union(count2_set)
        print(len(count_set))
    elif parts[0] == "1and1":
        count_set = set()
        count1_set,count1,data1 = queryTemp1(str(parts[1:4]), df)
        count2_set,count2,data2 = queryTemp1(str(parts[4:]), df)
        count_set = count1_set.intersection(count2_set)
        print(len(count_set))
    elif parts[0] == "1or2":
        count_set = set()
        count1_set,count1,data1 = queryTemp1(str(parts[1:4]), df)
        count2_set,count2,data2 = queryTemp2(str(parts[4:]), df)
        count_set = count1_set.union(count2_set)
        print(len(count_set))
    elif parts[0] == "1and2":
        count_set = set()
        count1_set,count1,data1 = queryTemp1(str(parts[1:4]), df)
        count2_set,count2,data2 = queryTemp2(str(parts[4:]), df)
        count_set = count1_set.intersection(count2_set)
        print(len(count_set))
    elif parts[0] == "2or2":
        count_set = set()
        count1_set,count1,data1 = queryTemp2(str(parts[1:3]), df)
        count2_set,count2,data2 = queryTemp2(str(parts[3:]), df)
        count_set = count1_set.union(count2_set)
        print(len(count_set))
    elif parts[0] == "2and2":
        count_set = set()
        count1_set,count1,data1 = queryTemp2(str(parts[1:3]), df)
        count2_set,count2,data2 = queryTemp2(str(parts[3:]), df)
        count_set = count1_set.intersection(count2_set)
        print(len(count_set))

    return count_set, len(count_set)


def main():
    '''
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


    #single_candidate = [[i] for i in single_candidate]
    print('number of length-1 frequent itemsets:\n'+ str(len(single_candidate)))
    data_list = []

    for i in range(len(Data)):
        row = list(Data_set.iloc[i])
        data_list.append(row)


    next_level_raw = list(itertools.combinations(single_candidate, 2))


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
                            Chart.loc[len(Chart)] = pd.Series({'RULE':str(Set_temp).strip('[]').replace('\'','')+','+item,'BODY':str(Set_temp).strip('[]').replace('\'',''),'HEAD':str(item),'CONFIDENCE':conf})
                        #print(str(conf))
                else:  ##如果候选项集个数超过了1
                    print("############"+str(i))
                    i_length_set = []
                    for x in range(len(previous)):
                        for y in range(x + 1, len(previous)):
                            L1 = previous[x][:i - 2]
                            L2 = previous[y][:i - 2]
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


                                conf = dict[str(all_frequentSet[h])]/dict[str(Set_temp)]

                                if conf >= int(confidence)/100:
                                    print(str(Set_temp) + "---->" + str(list_after_sort)+"conf is"+str(conf))
                                    counter = counter + 1
                                    i_length_set.append(list_after_sort)
                                    Chart.loc[len(Chart)] = pd.Series(
                                        {'RULE': str(Set_temp).strip('[]').replace('\'', '') + ',' + str(list_after_sort).strip('[]').replace('\'', ''),
                                         'BODY': str(Set_temp).strip('[]').replace('\'', ''), 'HEAD': str(list_after_sort).strip('[]').replace('\'', ''),
                                         'CONFIDENCE': conf})
                    print("num_"+str(i)+"is "+str(i_length_set))

                    previous = i_length_set
    print(counter)
'''
    # Chart.to_csv('Chart.csv',sep = ',')
    rule = input("Please input the rule with the following pattern\n asso_rule.template1(\"BODY\", 1, ['G59_Up', 'G10_Down'])\n")
    # asso_rule.template1("BODY", 1, ['G59_Up', 'G10_Down'])
    #asso_rule.template2("RULE", 3)
    #asso_rule.template3("1or1","BODY","ANY",['G10_DOWN'],"HEAD",1,['G59_Up'])
    df = pd.read_csv("Chart.csv")
    # print(df.RULE)
    queryOption(rule, df)

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
