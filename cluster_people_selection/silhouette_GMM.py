import random

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics, mixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from csv import reader

# read data

'''
# read MIT
file_name = 'mit.txt'
with open(file_name, 'r') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(data.shape)
    data = data[:, 1: -1]
'''

'''
# read STT
file_name = 'stt.txt'
with open(file_name, 'r') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x)
    print(data.shape)
    data = data[:, 1:]
    data = data.astype('float')
'''


# read HAR
file_name = 'har.txt'
with open(file_name, 'r') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(data.shape)
    data = data[:, 1: -1]


# clustering GMM
data = StandardScaler().fit_transform(data)

all_list = [i for i in range(len(data))]
random.seed(12)
a = random.sample(range(0, len(data)), int(len(data) / 2))
b = [i for i in all_list if i not in a]
print()

# har stratified clustering
a = [15,10,7,2,12,8,21,1,16,11,22] #[27,5,38,34,16,14,17,18,13,33,21,4,32,15,11,44,0,28,24,12,1,29,45]
b = [5,6,17,18,4,3,20,19,13,14] #[19,22,30,35,41,20,36,10,31,25,42,37,39,40,6,3,23,7,43,9,26,2,8]

'''
# MIT stratified clustering
a = [5,14,16,1,45,18,39,19,44,28,0,22,2,37,24,10,20,12,4] #[27,5,38,34,16,14,17,18,13,33,21,4,32,15,11,44,0,28,24,12,1,29,45]
b = [34,3,41,40,25,38,32,29,6,23,35,21,31,9,26,36,8,17,42] #[19,22,30,35,41,20,36,10,31,25,42,37,39,40,6,3,23,7,43,9,26,2,8]
print()
'''

grid_search = [3, 4, 5, 6, 7, 8]


for i in grid_search:
    gmm = mixture.GaussianMixture(n_components=i)
    aa = data[a]
    gmm.fit(aa)
    labels = gmm.fit_predict(aa)
    print('num_clusters:'+str(i))
    print(a)
    print(labels)
    a_quality = silhouette_score(aa, labels)
    print(str(a_quality))
    print('-------------------------------------')


for i in grid_search:
    gmm = mixture.GaussianMixture(n_components=i)
    bb = data[b]
    gmm.fit(bb)
    labels = gmm.fit_predict(bb)
    print('num_clusters:' + str(i))
    print(b)
    print(labels)
    b_quality = silhouette_score(bb, labels)
    print(str(b_quality))
    print('-------------------------------------')
    print('-------------------------------------')



'''
har two layer, each 4 clusters
 
[17, 9, 3, 19, 15, 22, 13, 20, 12, 0, 14, 4]
[1 1 0 0 0 1 3 1 3 2 3 1]
silhouette_score = 0.35087137647724703

-------------------------------------
[1, 2, 5, 6, 7, 8, 10, 11, 16, 18, 21, 23]
[1 0 1 0 0 3 1 1 1 0 3 2]
silhouette_score = 0.4033872832527085
'''

'''
mit two layer, each 7 clusters

[7, 14, 24, 12, 45, 18, 25, 9, 11, 21, 38, 23, 41, 10, 4, 42, 22, 5, 36, 35, 1, 34, 6]
[5 0 0 1 4 5 3 4 2 6 5 0 3 7 1 1 0 2 5 0 1 6 0]
silhouette_score = 0.6948058712307382
-------------------------------------
[0, 2, 3, 8, 13, 15, 16, 17, 19, 20, 26, 27, 28, 29, 30, 31, 32, 33, 37, 39, 40, 43, 44]
[5 2 6 1 4 5 5 0 7 6 1 2 4 1 3 4 1 7 3 5 2 2 1]
silhouette_score = 0.5769560143891007
'''
