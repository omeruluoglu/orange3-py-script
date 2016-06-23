import Orange
from Orange.clustering import kmeans, hierarchical
from Orange.data import Table
from Orange.evaluation import clustering
from Orange.distance import *
from Orange.widgets.evaluate import owconfusionmatrix
from collections import defaultdict

Hierarchical= hierarchical.HierarchicalClustering(n_clusters=3, linkage="average")                     
clusters= [Hierarchical]                     
#sample data operations
data = Table("iris.tab")
dist_matrix = Euclidean(data)
Hierarchical.fit(dist_matrix)

dict= defaultdict(int)
for label in Hierarchical.labels:
    #key=label, value=number of label
    dict[label] += 1
print("-----Clustering Results-----")  
for key,value in dict.items():
    print(key,value)
