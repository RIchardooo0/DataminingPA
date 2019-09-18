import pandas as pd
import itertools
import sys

geneData = pd.read_csv('associationruletestdata.txt', sep='\t', lineterminator='\n', header=None)#这行是抄的
#geneData.strip()
#print(geneData)

for i in range(len(geneData.columns)-1):
    geneData[i] = 'G' + str(i+1) + "_" + geneData[i].astype(str)  #这个也是抄的

print(geneData)

#G1 = []
#G1 =geneData.iloc[:,[0]]

#single_candidate单个的出现次数超过50的频繁项
single_candidate = set()

#print(G1)
#print(G1[0])

#geneData是100*100的dataFrame
count = 0
for j in range(1,101):
    print(j)
    count_up =0
    count_down=0
    for i in range(0,100):
   # print(geneData[0][i])
        k = j-1
        if geneData[k][i]=="G"+str(j)+"_Up":   #第j-1列第i行的数
           count_up = count_up +1
        if geneData[k][i]=="G"+str(j)+"_Down":   #第j-1列第i行的数
           count_down = count_down +1

   # print(count)
    str1 = "G" + str(j) + "_Up"
    str2 = "G" + str(j) +"_Down"
    if count_up>=30:  #如果UP出现次数>=50，那么set里就添加str1
        single_candidate.add(str1)
    if count_down>=30:
        single_candidate.add(str2)

#print("**********")
#print(len(single_candidate))


total_list = []
candidate = list(single_candidate)
list2 = list(itertools.combinations(candidate,2))

#print("**********")
#print(list2)

for j in range(0,100):
    print(j)
    list_son = []
    for i in range(0,100):
        list_son.append(geneData[i][j])
    total_list.append(list_son)

print("%%%%%%%%%%")
print(total_list)
dict = {}

for item_son in list2:
      count = 0

      #count2 = 0
      #print("**********")
      #print(item_son)
      for item_father in total_list:
       #count2+=1
       #print(count2)
       #print("!!!!!")
       #print(set(item_son))
       #print(set(item_father))
       if set(item_son).issubset(set(item_father)):
           count = count+1

      #print(count)

      if count>=30:
         # print("^^^^^^^")
          dict[str(set(item_son))]=count


print(len(dict))










print("&&&&&&&&&")
print(total_list)




#print(single_candidate)
#print(len(single_candidate))


#print(candidate)


#print(list2)

#set2 = set(list2)
#print(len(set2))

