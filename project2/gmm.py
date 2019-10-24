import pandas as pd
import numpy as np
import sys

data = np.loadtxt('cho.txt',delimiter='\t')
print(data)
#data = np.loadtxt('iyer.txt',delimiter='\t')
#df = pd.DataFrame(data)
label = data[:,1]
clusters = set((data[:,1]))
clusters.discard(-1)
K = len(clusters)
attributes = data[:,2:]
# X = attributes.mean(0)
print(attributes.shape)
print(attributes[1])
print(K)

def gmm_init(attributes, K):
    col = attributes.shape[1]
    pi = np.random.rand(K)
    pi = pi/sum(pi)
    means = []
    covs = []
    for i in range(K):
        mean = np.random.rand(col)
        means.append(mean)
        cov = np.random.rand(col,col)
        covs.append(cov)
    prob_matrix = np.zeros((attributes.shape[0], K))
    return pi, means, covs, prob_matrix

def gaussian(attributes, pi, means, covs, prob_matrix):
    K = len(means)
    for k in range(K):
        for row in range(attributes.shape[0]):
            dif = (attributes[row] - means[k])
            covdet = np.linalg.det(covs[k])
            covinv = np.linalg.inv(covs[k])
            prob = 1.0/(np.power(2*np.pi*np.abs(covdet),0.5)*np.exp(-0.5*dif.dot(covinv).dot(dif.T)))
            prob_matrix[row][k] = prob
    return prob_matrix

def s_step(pi, prob_matrix, attributes):
    K = len(pi)
    r = np.zeros((attributes.shape[0], K))
    for row in range(attributes.shape[0]):
        for k in range(K):
            numerator = prob_matrix[row][k]*pi[k]
            denominator = np.sum(pi*prob_matrix[row])
            r[row][k] = numerator/denominator
    return r

def m_step(pi, prob_matrix, attributes, r, means, covs):
    n = attributes.shape[0]
    cols = attributes.shape[1]
    pi_new = [0]*len(pi)
    covs_list = []
    u_list = []
    for k in range(K):
        pi_new[k] = np.sum(r[k])/n
    for k in range(K):
        u = np.zeros((1,cols))
        for row in range(n):
            temp = (r[row][k]*attributes[row])
            u = u + temp
        u_list.append(u/np.sum(r[k]))
    for k in range(K):
        cov = np.zeros((cols,cols))
        for row in range(n):
            temp = (r[row][k]*((attributes[row] - u_list[k]).T).dot(attributes[row] - u_list[k]))
            cov = cov + temp
        covs_list.append(cov/np.sum(r[k]))
    return covs_list, u_list, pi_new

K = 5
pi, means, covs, prob_matrix = gmm_init(attributes, K)
prob_matrix = gaussian(attributes, pi, means, covs, prob_matrix)
old_pi = np.array(pi)
old_means = np.array(means)
old_covs = np.array(covs)
r = s_step(pi, prob_matrix, attributes)
covs_list, u_list, pi_new = m_step(pi, prob_matrix, attributes, r, means, covs)
new_pi = np.array(pi_new)
new_means = np.array(u_list)
new_covs = np.array(covs_list)
b = 0.00000000001
while (np.sum(np.abs(new_pi - old_pi)) > b and np.sum(np.abs(new_covs.flatten() - old_covs.flatten()))> b and np.sum(np.abs(new_means - old_means)) > b):
    old_pi = np.array(pi_new)
    old_means = np.array(u_list)
    old_covs = np.array(covs_list)
    r = s_step(pi_new, prob_matrix, attributes)
    covs_list, u_list, pi_new = m_step(pi_new, prob_matrix, attributes, r, u_list, covs_list)
    new_pi = np.array(pi_new)
    new_means = np.array(u_list)
    new_covs = np.array(covs_list)

labels = [np.argmax(r[i]) for i in range(attributes.shape[0])]
print(labels)
print(sum(labels == label.T)/label.shape[0])

from sklearn.mixture import GaussianMixture 
data = np.loadtxt('cho.txt',delimiter='\t') 
df = pd.DataFrame(data[:,2:])
gmm = GaussianMixture(n_components = 5) 
gmm.fit(df) 
labels = gmm.predict(df)
print(labels)
print(sum(labels == label.T)/label.shape[0])

