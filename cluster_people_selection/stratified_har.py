import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics, mixture, model_selection
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from csv import reader


# read HAR
file_name = 'har.txt'
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
) = model_selection.train_test_split(XX, YY, indices, test_size=0.5, random_state=24, stratify=YY)


print()

'''
# stratified with age + sexy

[15, 10, 7, 2, 12, 8, 21, 1, 16, 11, 22]
[1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 1]
0.4147932634167539

[5, 6, 17, 18, 4, 3, 20, 19, 13, 14]
[2, 0, 0, 0, 0, 2, 0, 2, 1, 1]
0.37662394713054786





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