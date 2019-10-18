import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

file = 'cho.txt'
# file = 'iyer.txt'

df = pd.read_csv(file, sep='\t', lineterminator='\n', header=None)


data = df.loc[:,2:].values
y = df.loc[:,1].values

x= StandardScaler().fit_transform(data)

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
#
principalDF = pd.DataFrame(data = principalComponents,  columns = ['principal component 1', 'principal component 2'])
groundtruth = pd.DataFrame(data = df.loc[:,1].values, columns = ['Label'])
finalDf = pd.concat([principalDF, groundtruth], axis = 1)

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')

my_resutl = pd.DataFrame(data=np.array(cluster.fit_predict(data))+1, columns=['Label'])
my_Df = pd.concat([principalDF, my_resutl], axis=1)

fig = plt.figure(figsize=(8, 8))
bx = fig.add_subplot(1, 1, 1)
bx.set_xlabel('Principal Component 1', fontsize=15)
bx.set_ylabel('Principal Component 2', fontsize=15)
bx.set_title('2 Component PCA', fontsize=20)

# targets = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# colors = ['r', 'g', 'b', 'm', 'y', 'c','fuchsia', 'yellow', 'cyan', 'lime']

targets = [1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'm', 'y']

for target, color in zip(targets, colors):
    indicesToKeep = my_Df['Label'] == target
    bx.scatter(my_Df.loc[indicesToKeep, 'principal component 1']
               , my_Df.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
bx.legend(targets)
bx.grid()
plt.show()
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 Component PCA', fontsize = 20)
#
# targets = [-1,1,2,3,4,5,6,7,8,9,10]
# colors = ['r', 'g', 'b','m', 'y', 'fuchsia','yellow','cyan','lime']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['Label'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()


