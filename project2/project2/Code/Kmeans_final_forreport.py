import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import copy
import sys
from sklearn.cluster import SpectralClustering, KMeans

###########################################################
file = sys.argv[1]
data = np.loadtxt(file,delimiter='\t')
#data = np.loadtxt('iyer.txt',delimiter='\t')
#df = pd.DataFrame(data)
# print(data)


matrix = data[:,2:]
# print(matrix)

num_of_line,b = data.shape
# print(num_of_line)
# print(b)
num_of_vector = b-2

f_result = data[:,1:2]
final =[]
for i in range(0,num_of_line):
    final.append(float(f_result[i][0]))

############################################################
####     num_of_line    ####################################
####     num_of_vector  ####################################
############################################################



############################################################
####     initialize the cluster center #####################
############################################################

###########################################################################################
########################     initialize the cluster center     ############################
###########################################################################################
range_every_vec = []
min_every_vec = []
for j in range(num_of_vector):
    min_of_col_j = min(matrix[:,j])
    max_of_col_j = max(matrix[:,j])
    # print(min_of_col_j)
    # print(max_of_col_j)
    # print(float(max_of_col_j-min_of_col_j))
    min_every_vec.append(min_of_col_j)
    range_every_vec.append((max_of_col_j*10000-min_of_col_j*10000)/10000)
# print(range_every_vec)

rand = np.random.rand(num_of_vector, 1)
rand_forcal = rand.T.flatten()

# print(rand_forcal)
center = []

cluster_num =int(sys.argv[2])

for i in range(cluster_num):
    rand = np.random.rand(num_of_vector, 1)
    rand_forcal = rand.T.flatten()
    center_sub = []
    center_sub = min_every_vec+range_every_vec*rand_forcal
    center.append(list(center_sub))


#file = 'new_dataset_1.txt'
###############################################################################
###############################################################################
# cluster_num = 3 #the cluster number
# cluster_num = 10
# cluster_num = 10#for iyer
################################################################################
################################################################################
#center = [matrix[3,:],matrix[5,:],matrix[9,:]]
################################################################################
################################################################################
# print(center)

#######################################################################################
##################
belongto_which_cluster_list=np.ones(num_of_line)
dist_between_point_and_center =np.zeros(cluster_num)


#old_center = [[0 for i in range(num_of_vector)] for j in range(cluster_num)]
#old_belongto_which_cluster_list = np.zeros(num_of_line)
#####################求出每个点属于哪个中心####################
#while np.any(old_belongto_which_cluster_list != belongto_which_cluster_list):
for cishu in range(0,10):
    #print(cishu)

    #old_belongto_which_cluster_list = copy.deepcopy(belongto_which_cluster_list)
    # print(old_belongto_which_cluster_list)
    for line in range(0,num_of_line):
        #print(line)
        #dist_between_point_and_center = []
        for center_num in range(0,cluster_num):

            vec1 = np.array(matrix[line])
            vec2 = np.array(center[center_num])
            dist_between_point_and_center[center_num] = np.linalg.norm(vec1 - vec2)
            #print(dist_between_point_and_center[center_num])

        min_dis = max(dist_between_point_and_center) #initialize
        belong_to_which_center_temp = cluster_num   #initialize

        #print(dist_between_point_and_center[4])
        #print(type(dist_between_point_and_center))
        for iter in range(0,cluster_num):
            if dist_between_point_and_center[iter] < min_dis:
                min_dis = dist_between_point_and_center[iter]
                belong_to_which_center_temp = iter
        belongto_which_cluster_list[line] = belong_to_which_center_temp
    # print("@@@@@@@2")
    # #########################更新聚类中心#########################
    # print(old_belongto_which_cluster_list)
    # print(belongto_which_cluster_list)


    counter = np.zeros(cluster_num)
    center = np.zeros((cluster_num,num_of_vector))

    for line in range(0,num_of_line):
        for i in range(0,cluster_num):
            if belongto_which_cluster_list[line]==i:
                center[i]=list(np.array(center[i])+np.array(matrix[line]))
                counter[i] = counter[i]+1
                break
    # print(counter)
    for category_num in range(0, cluster_num):
        if counter[category_num]==0:
            rand_point_num = random.randint(0,num_of_line-1)
            center[category_num] = matrix[rand_point_num]
        else:
            for vector in range(0, num_of_vector):
                  center[category_num][vector] = center[category_num][vector]/counter[category_num]

    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #
    # print(old_belongto_which_cluster_list)
    # print(belongto_which_cluster_list)



'''      
total_list = []
for j in range(0,cluster_num):
    list = []
    for i in range(0,num_of_line):
        if belongto_which_cluster_list
'''



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


print("######################## final result #########################")

print(belongto_which_cluster_list)####################这个是最终的结果

truth = incidence_mat_gen(final)
result = incidence_mat_gen(belongto_which_cluster_list)

rand1,jaccard = ja_rand_cal(truth, result)

print("jaccard is"+str(jaccard))
print("rand is"+str(rand1))

##
#file = "iyer.txt"
# file = "cho.txt"
# file = sys.argv[1]
belongto_which_cluster_list = np.array(belongto_which_cluster_list) + 1

data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 2:])

ground_truth = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 1])

id = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 0])

X = data - data.mean(0)
# x = StandardScaler().fit_transform(x)
# print(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

principalDF = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
groundtruth = pd.DataFrame(data=ground_truth, columns=['Label'])
finalDf = pd.concat([principalDF, groundtruth], axis=1)

#DBSCAN result
my_resutl = pd.DataFrame(data = np.array(belongto_which_cluster_list), columns = ['Label'])
my_Df = pd.concat([principalDF, my_resutl ],axis = 1)

fig = plt.figure(figsize=(16, 8))
bx = fig.add_subplot(1, 2, 2)
bx.set_xlabel('Principal Component 1', fontsize=15)
bx.set_ylabel('Principal Component 2', fontsize=15)
bx.set_title('Kmeans Result on '+file, fontsize=20)
# bx.set_title('DBSCAN Result on iyer.txt', fontsize=20)

targets = [ i for i in range(1,cluster_num+1)]
colors = ['#' +''.join([random.choice('0123456789ABCDEF') for x in range(6)]) for i in range(cluster_num)]

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


targets = [ i for i in range(1,cluster_num+1)]
colors = ['#' +''.join([random.choice('0123456789ABCDEF') for x in range(6)]) for i in range(cluster_num)]

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()

# plt.savefig('Kmeans_iyer.eps')
# plt.savefig('Kmeans_cho.eps')
plt.show()
#print(belongto_which_cluster_list)
#print(final)