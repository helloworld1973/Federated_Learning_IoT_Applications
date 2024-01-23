import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics, mixture, model_selection
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from csv import reader

# read MIT
file_name = 'mit.txt'
with open(file_name, 'r') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(data.shape)
    XX = data[:, 1:-1]
    YY = data[:, -1:]

X_train, X_test, y_train, y_test, = model_selection.train_test_split(XX, YY, test_size=0.5, random_state=42, stratify=YY)

indices = np.arange(len(data))
(
    data_train,
    data_test,
    labels_train,
    labels_test,
    indices_train,
    indices_test,
) = model_selection.train_test_split(XX, YY, indices, test_size=0.5, random_state=1024, stratify=YY)


print()

'''
# stratified with age + sexy
[5,14,16,1,45,18,39,19,44,28,0,22,2,37,24,10,20,12,4]
[2 0 0 1 1 1 0 3 3 2 0 0 1 0 0 3 1 1 1]
0.6882175190538096


[34,3,41,40,25,38,32,29,6,23,35,21,31,9,26,36,8,17,42]
[0 1 3 1 3 1 0 0 0 0 0 0 2 2 0 1 0 2 1]
0.664622281335371
'''










'''
num_clusters:4
[27,5,38,34,16,14,17,18,13,33,21,4,32,15,11,44,0,28,24,12,1,29,45]
[0 2 0 1 1 1 3 0 3 1 1 0 1 1 2 1 1 2 1 0 0 1 3]
0.7253862639015771
'''

'''
num_clusters:4
[19,22,30,35,41,20,36,10,31,25,42,37,39,40,6,3,23,7,43,9,26,2,8]
[0 0 3 0 0 1 1 0 2 0 1 3 0 1 0 1 0 1 1 2 0 1 0]
0.5562913782929296
'''