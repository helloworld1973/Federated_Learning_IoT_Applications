import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics, mixture
from sklearn.datasets import make_blobs
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
    data = data[:, 1:]
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

#'''
# read HAR
file_name = 'har.txt'
with open(file_name, 'r') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(data.shape)
    data = data[:, 1:]
#'''

# clustering GMM
data = StandardScaler().fit_transform(data)

# clustering GMM
bicr = 10000000000
c = 0
slist = [5, 6]
gmm = mixture.GaussianMixture(n_components=slist[c], covariance_type='diag')
gmm.fit(data)
labels = gmm.fit_predict(data)
print(labels)
print(str(slist[c]) + ':' + str(bicr))
while (c < len(slist)):  # and (gmm.bic(data) < bicr)
    c += 1
    bicr = gmm.bic(data)
    gmm = mixture.GaussianMixture(n_components=slist[c], covariance_type='diag')
    gmm.fit(data)
    labels = gmm.fit_predict(data)
    print(labels)
    print(str(slist[c]) + ':' + str(bicr))

c = c - 1
gmm = mixture.GaussianMixture(n_components=slist[c], covariance_type='diag')
gmm.fit(data)
labels = gmm.fit_predict(data)
print()
