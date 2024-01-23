import os
from jinja2 import Environment, FileSystemLoader

# ====================================================================================================================
# all
# path = "./all_stratified_sampling/"

user_list = ['sub_' + str(i + 1) for i in range(24)]

#cluster_list = [-1, 0, 1, 2, 1, 0, 3, 1, 2, 3, 0, 2, 2, 2, 2, 1, 0, 1, 1, 0, 0, 2, 1, -1]  # manual selection 0: M 20-30, 1: F 20-30, 2: M 30-40, 3: F 30-40

#cluster_list = [-1, 00, 00, 11, 00, 00, 11, 00, 11, 11, 00, 11, 11, 11, 11, 00, 00, 00, 00, 00, 00, 11, 00, -1]  # manual selection 0: M 20-30, 1: F 20-30, 2: M 30-40, 3: F 30-40

# DBSCAN # cluster_list = [-1, 0, 1, -1, 1, 0, -1, 1, 2, 3, 0, -1, 4, 4, 4, -1, 0, 1, 3, 0, -1, 2, -1, -1]
cluster_list = [-1, 0, 1, -1, 1, 0, -1, 1, 2, 3, 0, -1, 4, 4, 4, -1, 0, 1, 3, 0, -1, 2, 3, -1]
# ====================================================================================================================
print()

# get_index = [x for x, y in enumerate(cluster_list) if y != -1]
# path = "./manual_har_cluster_all" + "/"
get_index = [x for x, y in enumerate(cluster_list) if y != 11]
path = "./har_all" + "/"
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

for i in range(4):
    get_index = [x for x, y in enumerate(cluster_list) if y == i]
    path = "./manual_har_cluster_" + str(i) + "/"
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
# ====================================================================================================================
