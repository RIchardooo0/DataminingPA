import pandas
import numpy

import pandas as pd
import numpy as np
import random

data = np.loadtxt('new_dataset_1.txt',delimiter='\t')
#data = np.loadtxt('iyer.txt',delimiter='\t')
#df = pd.DataFrame(data)
print(data)
print(type(data))
matrix1 = data[:,2:]

f_result = data[:,1:2]
final = []

print(f_result)
matrix = matrix1.tolist()
print(matrix)
geshu,total_weidu = data.shape

for i in range(0,geshu):
    final.append(int(f_result[i][0]))

weidu = total_weidu-2
print(geshu)
print(weidu)

##########this is x's neighbor
##############初始化###########
redius = 0.85
threshold = 4
################
def locate_in_matrix(point,total_matrix):
    #print("######")
    located_point = []

    for i in range(0,geshu):
        #print("point is"+str(point))
        #print("specific_point in matrix is" + str(list(total_matrix[i])))

        if point==total_matrix[i]:
            #print("point is"+str(point))
            #print("specific_point in matrix is" + str(list(total_matrix[i])))
            #print(i)
            located_point.append(i)
    return located_point

def find_neighbor(redius,point,total_matrix):
    neighbor=[]
    for i in range(0,geshu):
        vec1 = np.array(total_matrix[point])
        vec2 = np.array(total_matrix[i])
        #print("****")
        #print(vec1)
        #print(vec2)
        dist = np.linalg.norm(vec1 - vec2)
        #print(dist)
        #print("****")
        if dist<=redius:
        #if dist<=redius:
            neighbor.append(i)
    return neighbor

##matrix is the after_cleaning dataset
##geshu is number of the data, weidu is the dimension of the data

visited = []
not_visited = []

## ini: all of the point is not visited
for i in range(0,geshu):
    not_visited.append(i)
print(not_visited)

clus_num = 0;
clus_record = []
for i in range(0,geshu):
   clus_record.append(0)
#clus_record = np.zeros(geshu)

print(clus_record)

while len(not_visited)!=0:
    choice1 = random.randint(0, len(not_visited)-1)
    choice = not_visited[choice1]
    print("&&&"+str(choice))
    #print("&&&"+str(locate_in_matrix(not_visited[choice],matrix)))

    #selected_point = not_visited[choice]
    visited.append(choice)  # append it in visited list
    print(not_visited)
    print("********"+str(len(not_visited)))
    not_visited.remove(choice)  # delete it in not_visited list
    print("********"+str(len(not_visited)))
    print(not_visited)

    neighbor = find_neighbor(redius,choice,matrix)
    print("len of neighbor =" + str(len(neighbor)))
    if len(neighbor)>=threshold: #如果它的邻居大于了门槛
        clus_num = clus_num+1
        #point_loca = locate_in_matrix(selected_point,matrix)
        clus_record[choice]=clus_num
        print(clus_num)
        for item in neighbor: # for N中的每个点 p'
            if item not in visited: #如果 p'是unvisited
                visited.append(item)  # append it in visited list
                not_visited.remove(item)  # delete it in not_visited list

                neighbor1 = find_neighbor(redius, item, matrix)

                if len(neighbor1)>=threshold: #如果它的点数至少是threshold的点
                    for xiaxian in neighbor1: #让它发展一波下线
                        if xiaxian not in neighbor:
                            neighbor.append(xiaxian)

                #point_locat = locate_in_matrix(item,matrix)
                if clus_record[item]==0:# if p' doesn't belong to any cluster
                    clus_record[item]=clus_num  #把 p'添加到C
    else:
        clus_record[choice] = 0

    print(len(not_visited))


print(clus_record)
print(final)
















