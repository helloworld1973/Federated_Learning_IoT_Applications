import os
from jinja2 import Environment, FileSystemLoader

# ====================================================================================================================
# all
# path = "./all_stratified_sampling/"

user_list = [i for i in range(100, 103)] + [i for i in range(104, 110)] + [i for i in range(111, 120)] + \
             [i for i in range(121, 125)] + [i for i in range(200, 204)] + [i for i in range(205, 206)] + \
             [i for i in range(207, 211)] + [i for i in range(212, 216)] + [i for i in range(217, 218)] + \
             [i for i in range(220, 224)] + [i for i in range(228, 229)] + [i for i in range(230, 235)]

# cluster_list = [-1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, -1]  # GMM cluster 2
# cluster_list = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]

# cluster_list = [-1, 4, 0, 4, 0, 4, 0, 0, 3, 0, 4, 1, 1, 1, 1, 0, 4, 0, 0, 4, 2, 3, 0, -1]  # GMM cluster 4
# cluster_list = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1]

cluster_list = [3, 7, 0, 5, 7, 2, 3, 0, 3, 4,
                1, 2, 7, 4, 3, 3, 3, 5, 0, 1,
                5, 6, 3, 3, 3, 1, 3, 0, 2, 3,
                6, 4, 3, 1, 6, 3, 0, 6, 0, 3,
                0, 1, 7, 7, 1, 5]  # GMM cluster 8
#cluster_list = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1]

# ====================================================================================================================
print()
#'''
get_index = [x for x, y in enumerate(cluster_list) if y != -1]
path = "./mit_all" + "/"
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

#'''
'''
for i in [0, 1, 2, 3, 4, 5, 6, 7]:
    get_index = [x for x, y in enumerate(cluster_list) if y == i]
    path = "./GMM_8_clusters_mit_cluster_" + str(i) + "/"
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
# ====================================================================================================================
