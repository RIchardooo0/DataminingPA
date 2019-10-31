*********************
****  Datasets   ****
*********************
cho.txt and iyer.txt are the two datasets to be used for Project 2. 


*********************
**  Dataset format **
*********************

Each row represents a gene:
1) the first column is gene_id.
2) the second column is the ground truth clusters. You can compare it with your results. "-1" means outliers.
3) the rest columns represent gene's expression values (attributes).

***********************
1.GMM clustering algorithm:

GMM_iyer.py : 
This python file is responsible for clustering the file of iyer.txt. 
Code running : python GMM_iyer.py <dataset> <pi, means, covariance, iteration times, termination case (optional provided)>

GMM_cho.py: 
This python file is responsible for clustering the file of cho.txt.
Code running : python GMM_cho.py <dataset> <pi, means, covariance, iteration times, termination case (optional provided)>

2.Spectral clustering:
Spectral.py takes three arguments <filename> <clusternum> <sig value>


3.Hierarchical Clustering:
hierarchy.py takes two arguments <filename> <k cluster num>


4.DBSCAN cluster algorithm:
Code running: python DBSCAN_final <filename> <real k number><redius> <number_threshold>
The real k number is actual cluster of the dataset( if doesn’t have this, we can not draw the graph, but the algorithm doesn’t need this)

5.K-means algorithm:
Code running: python Kmeans_final_fordemo.py <filename> <the number of cluster>


