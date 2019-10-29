import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn import metrics
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

# file = "iyer.txt"
file = "cho.txt"
# file = sys.argv[1]

data = np.array(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 2:])

ground_truth = list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:, 1])

y_pred = SpectralClustering(n_clusters=5, gamma=0.2).fit_predict(data)

# print("Calinski-Harabasz Score", metrics.calinski_harabasz_score(data, y_pred))
# for index, gamma in enumerate((0.01,0.05,0.1,0.2,0.5,0.7,1,2)):
#     y_pred = SpectralClustering(n_clusters=5, gamma=gamma).fit_predict(data)
#     print ("Calinski-Harabasz Score with gamma=", gamma,metrics.calinski_harabasz_score(data, y_pred))

gen_result = y_pred + 1
print(gen_result)

inci_truth = incidence_mat_gen(ground_truth)
inci_hier = incidence_mat_gen(gen_result)

rand, jaccard = ja_rand_cal(inci_truth, inci_hier)
print(rand, jaccard)