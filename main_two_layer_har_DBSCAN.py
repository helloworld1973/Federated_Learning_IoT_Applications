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

user_list = ['sub_' + str(i + 1) for i in range(24)]

# GMM
cluster_a_index = [15, 8, 21, 16, 11, 4, 12, 0, 19, 7, 18, 10]
cluster_a_categories = [0, 1, 1, 2, 3, 4, 3, -1, 2, 4, 0, 2]

cluster_b_index = [1, 2, 3, 5, 6, 9, 13, 14, 17, 20, 22, 23]
cluster_b_categories = [0, 1, 0, 0, 1, 1, 0, 0, 1, -1, 1, 1]

'''
# ===========
file_name = 'har.txt'
with open(file_name, 'r') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print(data.shape)
    data = data[:, 1:]
data = StandardScaler().fit_transform(data)

a_list = []
for i in [0, 1, 2, 3, 4]:
    get_a_index = [x for x, y in enumerate(cluster_a_categories) if y == i]
    cluster_a_index_list = [cluster_a_index[j] for j in get_a_index]
    data_list = [data[j] for j in cluster_a_index_list]
    data_centre_point = np.mean(np.array(data_list), axis=0)
    a_list.append(data_centre_point)
a_list = np.array(a_list)

b_list = []
for i in [0, 1]:
    get_b_index = [x for x, y in enumerate(cluster_b_categories) if y == i]
    cluster_b_index_list = [cluster_b_index[j] for j in get_b_index]
    data_list = [data[j] for j in cluster_b_index_list]
    data_centre_point = np.mean(np.array(data_list), axis=0)
    b_list.append(data_centre_point)
b_list = np.array(b_list)

results = cdist(a_list, b_list)
print()
# min_list = np.min(results, axis=0)
print()
# 0->2  1->4
# 


# ====================================================================================================================
print()
'''

'''
# get_index = [x for x, y in enumerate(cluster_list) if y != -1]
# path = "./manual_har_cluster_all" + "/"
get_index = [x for x, y in enumerate(cluster_list) if y != -1]
path = "./DBSCAN_5_clusters_har_cluster_all" + "/"
os.makedirs(path, exist_ok=False)


env = Environment(loader=FileSystemLoader(searchpath=""))
template = env.get_template("./template_har.py")

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

left_list = [0, 1]
right_list = [2, 4]
'''
for i in [0, 1]:
    get_index_list = [x for x, y in enumerate(cluster_b_categories) if y == i]
    get_index = [cluster_a_index[j] for j in get_index_list]
    path = "./two_layer_clusters_har_DBSCAN_" + str(i) + "/"
    os.makedirs(path, exist_ok=False)


    env = Environment(loader=FileSystemLoader(searchpath=""))
    template = env.get_template("./template_har.py")

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

for i in [2, 4]:
    get_index_list = [x for x, y in enumerate(cluster_a_categories) if y == i]
    get_index = [cluster_a_index[j] for j in get_index_list]
    path = "./two_layer_clusters_har_DBSCAN_" + str(i) + "/"
    os.makedirs(path, exist_ok=False)


    env = Environment(loader=FileSystemLoader(searchpath=""))
    template = env.get_template("./template_har.py")

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

#'''
# ====================================================================================================================
