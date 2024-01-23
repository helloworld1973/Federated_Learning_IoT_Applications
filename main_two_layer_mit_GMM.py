import os
from scipy.spatial.distance import cdist
import torch
from jinja2 import Environment, FileSystemLoader
from csv import reader
from sklearn.preprocessing import StandardScaler
import numpy as np

# ====================================================================================================================
# all
# path = "./all_stratified_sampling/"

user_list = [i for i in range(100, 103)] + [i for i in range(104, 110)] + [i for i in range(111, 120)] + \
            [i for i in range(121, 125)] + [i for i in range(200, 204)] + [i for i in range(205, 206)] + \
            [i for i in range(207, 211)] + [i for i in range(212, 216)] + [i for i in range(217, 218)] + \
            [i for i in range(220, 224)] + [i for i in range(228, 229)] + [i for i in range(230, 235)]


# Stratified sampling GMM MIT
cluster_a_index = [27, 5, 38, 34, 16, 14, 17, 18, 13, 33, 21, 4, 32, 15, 11, 44, 0, 28, 24, 12, 1, 29, 45]
cluster_a_categories = [0, 2, 0, 1, 1, 1, 3, 0, 3, 1, 1, 0, 1, 1, 2, 1, 1, 2, 1, 0, 0, 1, 3]

cluster_b_index = [19, 22, 30, 35, 41, 20, 36, 10, 31, 25, 42, 37, 39, 40, 6, 3, 23, 7, 43, 9, 26, 2, 8]
cluster_b_categories = [0, 0, 3, 0, 0, 1, 1, 0, 2, 0, 1, 3, 0, 1, 0, 1, 0, 1, 1, 2, 0, 1, 0]

'''
# GMM
cluster_a_index = [30, 17, 42, 33, 22, 9, 24, 0, 23, 45, 44, 29, 14, 38, 40, 43, 11, 5, 10, 6, 1, 18, 26]
cluster_a_categories = [0, 2, 1, 4, 4, 2, 4, 4, 4, 2, 4, 4, 4, 1, 1, 1, 3, 3, 4, 4, 1, 1, 4]
# [1, 2, 3, 4]

cluster_b_index = [2, 3, 4, 7, 8, 12, 13, 15, 16, 19, 20, 21, 25, 27, 28, 31, 32, 34, 35, 36, 37, 39, 41]
cluster_b_categories = [4, 1, 1, 4, 0, 1, 5, 0, 0, 3, 1, 6, 3, 4, 2, 5, 0, 6, 0, 4, 6, 0, 7]
# [0, 1, 3, 4, 5, 6]
'''

'''
# ===========
# read MIT
file_name = 'mit.txt'
with open(file_name, 'r') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(data.shape)
    data = data[:, 1:]
data = StandardScaler().fit_transform(data)

a_list = []
for i in [0, 1, 2, 3]:
    get_a_index = [x for x, y in enumerate(cluster_a_categories) if y == i]
    cluster_a_index_list = [cluster_a_index[j] for j in get_a_index]
    data_list = [data[j] for j in cluster_a_index_list]
    data_centre_point = np.mean(np.array(data_list), axis=0)
    a_list.append(data_centre_point)
a_list = np.array(a_list)

b_list = []
for i in [0, 1, 2, 3]:
    get_b_index = [x for x, y in enumerate(cluster_b_categories) if y == i]
    cluster_b_index_list = [cluster_b_index[j] for j in get_b_index]
    data_list = [data[j] for j in cluster_b_index_list]
    data_centre_point = np.mean(np.array(data_list), axis=0)
    b_list.append(data_centre_point)
b_list = np.array(b_list)

results = cdist(a_list, b_list)
print()
# min_list = np.min(results, axis=0)

# 0->3  1->1  2->4  3->0
# 1->4  2->1  3->5  4->0
'''

# Stratified: 0->1  1->0  2->2  3->2


# ====================================================================================================================
print()

left_list = [0, 1, 2, 3]
right_list = [1, 0, 2, 2]

'''
for i in [0, 1, 2, 3]:
    get_index_list = [x for x, y in enumerate(cluster_b_categories) if y == i]
    get_index = [cluster_b_index[j] for j in get_index_list]
    path = "./two_layer_stratified_clusters_mit_GMM_" + str(i) + "/"
    os.makedirs(path, exist_ok=False)


    env = Environment(loader=FileSystemLoader(searchpath=""))
    template = env.get_template("./template_mit.py")

    index_list = [user_list[i] for i in get_index]


    for i in range(0, len(index_list)):
        output = template.render({'user_name': index_list[i]})
        with open(path + "client%d.py" % i, 'w') as out:
            out.write(output)
            out.close()


    env = Environment(loader=FileSystemLoader(searchpath=""))
    template = env.get_template("./run.sh")


    output = template.render({'num_users': len(index_list) - 1})
    with open(path + "run.sh", 'w') as out:
        out.write(output)
        out.close()
'''

for i in [0, 1, 2, 3]:
    get_index_list = [x for x, y in enumerate(cluster_a_categories) if y == i]
    get_index = [cluster_a_index[j] for j in get_index_list]
    path = "./two_layer_stratified_clusters_mit_GMM_" + str(i) + "/"
    os.makedirs(path, exist_ok=False)


    env = Environment(loader=FileSystemLoader(searchpath=""))
    template = env.get_template("./template_mit.py")

    index_list = [user_list[i] for i in get_index]


    for i in range(0, len(index_list)):
        output = template.render({'user_name': index_list[i]})
        with open(path + "client%d.py" % i, 'w') as out:
            out.write(output)
            out.close()


    env = Environment(loader=FileSystemLoader(searchpath=""))
    template = env.get_template("./run.sh")


    output = template.render({'num_users': len(index_list) - 1})
    with open(path + "run.sh", 'w') as out:
        out.write(output)
        out.close()

# ====================================================================================================================
