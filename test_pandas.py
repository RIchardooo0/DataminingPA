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
    count =0
    for i in range(0,99):
   # print(geneData[0][i])
        k = j-1
        if geneData[k][i]=="G"+str(j)+"_Up":   #第j-1列第i行的数
           count = count +1
   # print(count)
    str1 = "G" + str(j) + "_Up"
    str2 = "G" + str(j) +"_Down"
    if count>=50:  #如果UP出现次数>=50，那么set里就添加str1
        single_candidate.add(str1)
        #print(j)
    else:  #如果小于了50，那么就添加str2
        single_candidate.add(str2)
       # print(j)

print(single_candidate)
print(len(single_candidate))



#print("num ofcount)
#print(geneData[1:10])


#geneData[1] = 'G' + str(1 + 1) + "_" + geneData[1].astype(str)
#print(geneData[1])
#print(len(geneData.columns))

#frequent_itemsets = set()
#for i in range(len(geneData.columns)):
#    dat=geneData[i].groupby(geneData[i]).describe()
 #   print("dat")
  #  print(dat)
   # for j in range(len(dat)):
    #    item=list(dat.iloc[j])[2:4]
   #     if(item[1] >= 0.5):
#        frequent_itemsets.add(item[0])