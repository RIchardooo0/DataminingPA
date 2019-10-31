import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sys
import math
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.covariance
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


def gmm_init(attributes, K):
    col = attributes.shape[1]
    pi = np.ones(K) / K
#     pi = [0.5, 0.5]
    kmean = KMeans(n_clusters=K)
    kmean.fit(attributes);

    means = kmean.cluster_centers_
    
#     means = np.array([[0,0],[1,1]])
    covs = []
    for i in range(K):
        cov = np.eye(col)/2000
#         cov = sklearn.covariance.empirical_covariance(attributes, assume_centered=True)
        covs.append(cov)
#     cov1 = np.array([[2,1],[1,1]])
#     cov2 = np.array([[3,2],[2,2]])
#     cov3 = np.array([[4,3],[2,2]])
#     covs.append(cov1)
#     covs.append(cov2)
#     covs.append(cov3)
#     for i in range(K):
#             cov = np.random.rand(col,col)
#             covs.append(cov)
   
    prob_matrix = np.zeros((attributes.shape[0], K))
    return pi, means, covs, prob_matrix

def gaussian(attributes, pi, means, covs, prob_matrix):
    K = len(means)
    b = pow(10,-9)
    for k in range(K):
        for row in range(attributes.shape[0]):
            dif = (attributes[row,:] - means[k])
            covdet = np.linalg.det(covs[k])
            
#             covdet = abs(covdet)
            if covdet == 0:
                covdet = b
            covinv = np.linalg.pinv(covs[k])
            
            prob = (1.0/(pow((2*np.pi)*covdet,0.5)))*np.exp(-1*0.5*(dif.dot(covinv)).dot(dif.T))
            if (prob == 0):
                prob = b
            prob_matrix[row][k] = prob
          
    return prob_matrix

def e_step(pi, prob_matrix, attributes):
    K = len(pi)
    r = np.zeros((attributes.shape[0], K))
    for k in range(K):
        for row in range(attributes.shape[0]):
            numerator = prob_matrix[row][k]*pi[k]
            denominator = np.sum(np.multiply(pi,prob_matrix[row]))
            r[row][k] = numerator/denominator     
    return r

def m_step(pi, prob_matrix, attributes, r, means, covs):
    n = attributes.shape[0]
    cols = attributes.shape[1]
    pi_new = [0]*len(pi)
    covs_list = []
    u_list = []
    for k in range(K):
        pi_new[k] = np.sum(r[:,k])/n  
    for k in range(K):
        u = np.zeros((1,cols))
        for row in range(n):
            temp = (r[row][k]*attributes[row])
            u = u + temp
        u_list.append(u/np.sum(r[:,k]))
    for k in range(K):
        cov = np.zeros((cols,cols))
        for row in range(n):
            temp = (r[row][k]*((attributes[row] - u_list[k]).T).dot(attributes[row] - u_list[k]))
            cov = cov + temp
        covs_list.append(cov/np.sum(r[:,k])) 
    return covs_list, u_list, pi_new

# data = np.loadtxt('GMM_tab_seperated.txt',delimiter='\t')
data = np.loadtxt('iyer.txt',delimiter='\t')
# data = np.loadtxt('GMM.txt')
label = data[:,1]
clusters = set((data[:,1]))
clusters.discard(-1)
K = len(clusters)
attributes = data[:,2:]
pi, means, covs, prob_matrix = gmm_init(attributes, K)
prob_matrix = gaussian(attributes, pi, means, covs, prob_matrix)
old_pi = np.array(pi)
old_means = np.array(means)
old_covs = np.array(covs)
r = e_step(pi, prob_matrix, attributes)
covs_list, u_list, pi_new = m_step(pi, prob_matrix, attributes, r, means, covs)
new_pi = np.array(pi_new)
new_means = np.array(u_list)
new_covs = np.array(covs_list)
b = pow(10,-9)
time = 1
print(time)
print("==========================")
print(new_pi)
print(new_means)
print(new_covs)
print(r)
while (np.linalg.norm(new_covs - old_covs) > b):
    old_pi = np.array(new_pi)
    old_means = np.array(new_means)
    old_covs = np.array(new_covs)
    prob_matrix = gaussian(attributes, new_pi, new_means, new_covs, prob_matrix)
    r = e_step(new_pi, prob_matrix, attributes)
    covs_list, u_list, pi_new = m_step(new_pi, prob_matrix, attributes, r, new_means, new_covs)
    time = time + 1
    new_pi = np.array(pi_new)
    new_means = np.array(u_list)
    new_covs = np.array(covs_list)
    print(time)
    print("==========================")
    print(new_pi)
    print(new_means)
    print(new_covs)
    print(r)
    if (time == 6):
        break
labels = [np.argmax(r[i]) for i in range(attributes.shape[0])]
truth = incidence_mat_gen(label)
result = incidence_mat_gen(labels)
labels = np.array(labels)+1

rand1,jaccard = ja_rand_cal(truth, result)

print("Jaccard"+str(jaccard))
print("Rand"+str(rand1))


#PCA implementation
file = 'iyer.txt'

data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 2:])

ground_truth = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 1])
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

#Hierarchy result
my_resutl = pd.DataFrame(data = np.array(labels), columns = ['Label'])
my_Df = pd.concat([principalDF, my_resutl ],axis = 1)

fig = plt.figure(figsize=(16, 8))
bx = fig.add_subplot(1, 2, 2)
bx.set_xlabel('Principal Component 1', fontsize=15)
bx.set_ylabel('Principal Component 2', fontsize=15)
# bx.set_title('Hierarchical Clustering Result on Cho.txt', fontsize=20)
bx.set_title('GMM Clustering Result on iyer.txt', fontsize=20)

targets = [ i for i in range(1,int(K)+1)]
colors = ['#' +''.join([random.choice('0123456789ABCDEF') for x in range(6)]) for i in range(int(K))]

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


targets = [ i for i in range(1,int(K)+1)]
colors = ['#' +''.join([random.choice('0123456789ABCDEF') for x in range(6)]) for i in range(int(K))]

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
# plt.savefig('hierarchy_cho.eps')
# plt.savefig('GMM_iyer1.eps')
plt.show()

#
# from sklearn.mixture import GaussianMixture
# # data = np.loadtxt('GMM_tab_seperated.txt',delimiter='\t')
# data = np.loadtxt('iyer.txt',delimiter='\t')
# # data = np.loadtxt('GMM.txt')
# df = pd.DataFrame(data[:,2:])
# gmm = GaussianMixture(n_components = K)
# gmm.fit(df)
# labels = gmm.predict(df)
# truth = incidence_mat_gen(label)
# result = incidence_mat_gen(labels)
#
# rand1,jaccard = ja_rand_cal(truth, result)
#
# print("diaobao"+str(jaccard))
# print("diaobao"+str(rand1))



