import numpy as np
import pandas as pd
import random

data = pd.read_csv('project3_dataset2.txt', sep='\t', header = None)
data = data.replace(to_replace = 'Present', value = 1)
data = data.replace(to_replace = 'Absent', value = 0) 
labels = data.iloc[:,-1].value_counts().index
means = []
variances = []
res = []
for label in labels:
    data_label = data.loc[data.iloc[:,-1] == label,:]
    mean = data_label.iloc[:,:-1].mean()
    variance = np.sum((data_label.iloc[:,:-1]- mean)**2)/(data_label.shape[0])
    means.append(mean)
    variances.append(variance)
means = pd.DataFrame(means, index = labels)
variances = pd.DataFrame(variances, index = labels)
for i in range(data.shape[0]):
    temp = data.iloc[i,:-1] - means
    probability = 1
    prob = np.exp(-1*(temp)**2/(variances*2))/np.sqrt(2*np.pi*variances)
    for j in range(data.shape[1] - 1):
        probability = probability*prob[j]
    pre = np.argmax(probability.values)
    res.append(pre)
truth = np.array(data.iloc[:,-1])
num = sum(res == truth)
n = data.shape[0]
accuracy = num/n
print(accuracy)
