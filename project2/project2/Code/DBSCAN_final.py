import sys
import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

file = sys.argv[1]

data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 2:])

ground_truth = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 1])

id = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 0])

RealK = sys.argv[2]


matrix = data

f_result = ground_truth

geshu, weidu = data.shape

## jaccard and rand
def ja_rand_cal(truth, result):
    M11 = np.sum(truth*result)

    new_inci = truth + result
    count = 0
    for i in range(len(new_inci)):
        for j in range(len(new_inci[0])):
            if new_inci[i][j] == 0:
                count+=1
    M00 =count
    rand = (M11+M00)/(len(truth)**2)
    jaccard = M11/(len(truth)**2 - M00)

    return rand, jaccard

def incidence_mat_gen(label):
    matrix = [[0]*len(label) for i in range(len(label))]
    for i in range(len(label)):
        for j in range(i,len(label)):
            if label[i] == label[j]:
                matrix[i][j] = 1
                matrix[j][i] = 1
            else:
                matrix[i][j] = 0
                matrix[j][i] = 0
    return np.array(matrix)
# print(f_result)
##########this is x's neighbor
##############初始化###########
#for cho
# redius = 1
# threshold = 3

#for iyer
# redius = 1
# threshold = 5
################
redius = float(sys.argv[3])
threshold = float(sys.argv[4])

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
# print(not_visited)

clus_num = -1;
clus_record = [-1 for i in range(geshu)]
#clus_record = np.zeros(geshu)


while len(not_visited)!=0:
    is_valid = False
    while is_valid != True:
        choice = random.randint(0, geshu - 1)  # this is the suoyin in the
        if choice in not_visited and choice not in visited:
            is_valid = True
    # print("&&&"+str(choice))
    #print("&&&"+str(locate_in_matrix(not_visited[choice],matrix)))

    #selected_point = not_visited[choice]
    visited.append(choice)  # append it in visited list
    # print(not_visited)
    # print("********"+str(len(not_visited)))
    not_visited.remove(choice)  # delete it in not_visited list
    # print("********"+str(len(not_visited)))
    # print(not_visited)

    neighbor = find_neighbor(redius,choice,matrix)
    # print("len of neighbor =" + str(len(neighbor)))
    if len(neighbor)>=threshold: #如果它的邻居大于了门槛
        clus_num = clus_num+1
        #point_loca = locate_in_matrix(selected_point,matrix)
        clus_record[choice]=clus_num
        # print(clus_num)
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
            if clus_record[item]==-1:# if p' doesn't belong to any cluster
                    clus_record[item]=clus_num  #把 p'添加到C
    else:
        clus_record[choice] = -1

    # print(len(not_visited))
# print(clus_record)
k = len(set(clus_record))
# print(k)
# print(k)
#PCA implementation

# print(clus_record)
inci_truth = incidence_mat_gen(ground_truth)
inci_hier = incidence_mat_gen(clus_record)

rand, jaccard = ja_rand_cal(inci_truth, inci_hier)
print(rand, jaccard)

########################################################################################################################
new_clus = []
for i in clus_record:
    if i>=0:
        new_clus.append(i+1)
    else:
        new_clus.append(i)

# print(new_clus)
clus_record = np.array(new_clus)
# print(clus_record)
#######################################################################################################################
#PCA implementation
df = pd.read_csv(file, sep='\t', lineterminator='\n', header=None)

x = df.loc[:, 2:].values
X = x - x.mean(0)
# x = StandardScaler().fit_transform(x)
# print(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

principalDF = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
groundtruth = pd.DataFrame(data=df.loc[:, 1].values, columns=['Label'])
finalDf = pd.concat([principalDF, groundtruth], axis=1)

#Spectral result
my_resutl = pd.DataFrame(data = np.array(clus_record), columns = ['Label'])
my_Df = pd.concat([principalDF, my_resutl ],axis = 1)

fig = plt.figure(figsize=(16, 8))
bx = fig.add_subplot(1, 2, 2)
bx.set_xlabel('Principal Component 1', fontsize=15)
bx.set_ylabel('Principal Component 2', fontsize=15)
# bx.set_title('Spectral Clustering Result on cho.txt', fontsize=20)
bx.set_title('DBSCAN Result', fontsize=20)

targets = [ i for i in range(1,int(k)+1)]
targets.append(-1)
colors = ['#' +''.join([random.choice('0123456789ABCDEF') for x in range(6)]) for i in range(int(k)+1)]

for target, color in zip(targets, colors):
    indicesToKeep = my_Df['Label'] == target
    bx.scatter(my_Df.loc[indicesToKeep, 'principal component 1']
               , my_Df.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
bx.legend(targets)
bx.grid()
#Ground truth
#####################################
ax = fig.add_subplot(1, 2, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('Ground Truth', fontsize=20)




targets = [ i for i in range(1,int(RealK)+1)]
colors = ['#' +''.join([random.choice('0123456789ABCDEF') for x in range(6)]) for i in range(int(RealK))]

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
# plt.savefig('spectral_cho.eps')
# plt.savefig('spectral_iyer.eps')
# plt.savefig('DBSCAN_cho.eps')
# plt.savefig('DBSCAN_iyer.eps')
plt.show()