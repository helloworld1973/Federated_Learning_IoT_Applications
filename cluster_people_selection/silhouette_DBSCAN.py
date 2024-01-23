import random

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics, mixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from csv import reader

# read data

#'''
# read MIT
file_name = 'mit.txt'
with open(file_name, 'r') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(data.shape)
    data = data[:, 1:]
#'''

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

'''
# read HAR
file_name = 'har.txt'
with open(file_name, 'r') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(data.shape)
    data = data[:, 1:]
'''

# clustering GMM
data = StandardScaler().fit_transform(data)

all_list = [i for i in range(len(data))]
random.seed(12)
a = random.sample(range(0, len(data)), int(len(data) / 2))
b = [i for i in all_list if i not in a]
print()

eps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
min_samples_list = [1, 2, 3, 4]

for i in eps_list:
    for j in min_samples_list:
        try:
            aa = data[a]
            db = DBSCAN(eps=i, min_samples=j).fit(aa)  # 0.68 four clusters HAR   # 0.1 eight clusters MIT
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            print('eps:'+str(i)+'__min_samples:'+str(j))
            print(a)
            print(labels)
            a_quality = silhouette_score(aa, labels)
            print(str(a_quality))
            print('-------------------------------------')
        except:
            print(str(i) + '__' + str(j))
            print("can not even has one cluster")
            print('-------------------------------------')

for i in eps_list:
    for j in min_samples_list:
        try:
            bb = data[b]
            db = DBSCAN(eps=i, min_samples=j).fit(bb)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            print('eps:'+str(i)+'__min_samples:'+str(j))
            print(b)
            print(labels)
            b_quality = silhouette_score(bb, labels)
            print(str(b_quality))
            print('-------------------------------------')
            print('-------------------------------------')
        except:
            print(str(i) + '__' + str(j))
            print("can not even has one cluster")
            print('-------------------------------------')

'''
har two layer, each 3 clusters

[15, 8, 21, 16, 11, 4, 12, 0, 19, 7, 18, 10]
[-1  0  0  1 -1  2 -1 -1  1  2 -1  1]
silhouette_score = 0.2887524201946839
-------------------------------------
[1, 2, 3, 5, 6, 9, 13, 14, 17, 20, 22, 23]
[ 0  1 -1  0 -1 -1  2  2  1 -1 -1 -1]
silhouette_score = 0.2037482842069448
'''

'''
mit two layer, each 5 clusters

[30, 17, 42, 33, 22, 9, 24, 0, 23, 45, 44, 29, 14, 38, 40, 43, 11, 5, 10, 6, 1, 18, 26]
[-1 -1  0  1  1 -1  2  2  2 -1  1  1  2  3  3  0  4  4  1  1  0  3  1]
silhouette_score = 0.494405758587215
-------------------------------------
[2, 3, 4, 7, 8, 12, 13, 15, 16, 19, 20, 21, 25, 27, 28, 31, 32, 34, 35, 36, 37, 39, 41]
[ 0  1  1  0  2  1  3  2  2  4  1  2  4  0  3  3  2  2  2  0  2  2 -1]
silhouette_score = 0.6186597930397515

'''
