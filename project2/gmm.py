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
            dif = (attributes[row] - mean[k])
            covdet = np.linalg.det(covs[k])
            covinv = np.linalg.inv(covs[k])
            prob = 1.0/(np.power(np.power(2*np.pi*np.abs(covdet),0.5))*np.exp(-0.5*dif.dot(covinv).dot(dif.T)))
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
    u = means
    covs_list = []
    for k in range(K):
        pi[k] = np.sum(r[k])/n
        u[k] = np.sum((r[k].T).dot(attributes))/np.sum(r[k])#not sure
    for k in range(K):
        num = []
        denom = []
        for row in range(n):
            num.append(r[row][k]*(attributes[row] - u[k]).dot((attributes[row] - u[k]).T))
        covs_list.append(np.sum(num)/np.sum(r[k]))
    return
    
