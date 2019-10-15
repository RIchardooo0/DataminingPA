import pandas
import numpy

import pandas as pd
import numpy as np

data = np.loadtxt('cho.txt',delimiter='\t')
#data = np.loadtxt('iyer.txt',delimiter='\t')
#df = pd.DataFrame(data)
print(data)

matrix = data[:,2:]
print(matrix)

a,b = data.shape
print(a)
print(b)
vector_num = b-2

cluster_num = 5


center = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [-0.69,-0.96,-1.16,-0.66,-0.55,0.12,-1.07,-1.22,0.82,1.4,0.71,0.68,0.11,-0.04,0.19,0.82],
          [-0.378,0.86,1.14,-0.11,-0.32,-0.26,-0.84,-1.11,-0.24,1.23,0.54,0.02,0.048,-0.66,-0.47,-0.41],
          [-0.86,-1.26,0.17,0.56,1.23,0.66,0.31,-0.34,-0.28,-0.14,0.56,0.76,0.43,0.231,-0.01,-0.33],
          [-0.79,-1.04,-0.49,-0.12,0.61,0.77,0.34,0.12,0.12,-0.35,0.15,0.22,0.37,0.29,0.17,0.14],
          [-0.43,-0.84,-1.22,0.05,0.63,0.57,0.54,0.29,0.21,-0.11,0.16,0.62,0.14,0.25,0.157,0.0]]
distance =np.zeros(cluster_num+1)
temp_cate=np.zeros(a)
#每个类的第一个
for cishu in range(0,20):
       for line in range(0,a):
             for cate in range(1,6): #左闭右开
                 #print(cate)
                 for vector in range(0,vector_num):
                     distance[cate] = distance[cate] + (center[cate][vector]-matrix[line][vector])*(center[cate][vector]-matrix[line][vector])
                     #print(center[cate])
                     #print(distance[cate])
             min = 10000
             min_num = 100
             #print(distance)
             for iter in range(1,6):
                 if distance[iter]<min:
                     min = distance[iter]
                     min_num = iter
             temp_cate[line] = min_num
             #print(temp_cate)
             distance = np.zeros(cluster_num + 1)
             #print(distance)

       center = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
       print(temp_cate)
       num_of_each_clustor = np.zeros(cluster_num+1) #make it 1-5
       for line in range(0,a):  #now all the point has their cluster
           #print(line)
           for category_num in range(1,cluster_num+1):
               if temp_cate[line]==category_num:
                   #print(category_num)
                   for vector in range(0,vector_num):
                       center[category_num][vector] = center[category_num][vector] + matrix[line][vector]
                   num_of_each_clustor[category_num] = num_of_each_clustor[category_num]+1
                   break

       #print(num_of_each_clustor)
       #print(center)
       for category_num in range(1, cluster_num + 1):
           for vector in range(0, vector_num):
               center[category_num][vector] = center[category_num][vector]/num_of_each_clustor[category_num]


       #print(center)
       #print(cishu)
       #print(temp_cate)







