from glob import glob
from gtda.time_series import SlidingWindow

Segment_Size = 128
activity_codes = {'dws': 0, 'jog': 1, 'sit': 2, 'std': 3, 'ups': 4, 'wlk': 5}
activity_types = list(activity_codes.keys())

def load_data(num):
    # Load All data:
    Folders = glob('./A_DeviceMotion_data/*_*')
    Folders = [s for s in Folders if "csv" not in s]

    X = []
    y = []
    for j in Folders:
        Csv = glob(j + '/'+num+'.csv')[0]
        label = int(activity_codes[j[22:25]])
        x_a_activity_a_user_complete_data = []
        y_a_activity_a_user_complete_data = []
        with open(Csv, 'r') as f:
            lines = f.readlines()
            for num_index, line in enumerate(lines):
                if num_index != 0:
                    a_row_column = line.replace('\n', '').split(',')
                    new_a_row_column = []
                    for index, i in enumerate(a_row_column):
                        if index != 0:
                            new_a_row_column.append(float(i))

                    x_a_activity_a_user_complete_data.append(new_a_row_column)
                    y_a_activity_a_user_complete_data.append(label)

        sliding_bag = SlidingWindow(size=Segment_Size, stride=int(Segment_Size/2))
        X_bags = sliding_bag.fit_transform(x_a_activity_a_user_complete_data)
        y_bags = [label for i in range(len(X_bags))]

        for a_X_bag in X_bags:
            X.append(a_X_bag)
        for a_y_bag in y_bags:
            y.append(a_y_bag)

    return X, y

classes = {'dws': 0, 'jog': 1, 'sit': 2, 'std': 3, 'ups': 4, 'wlk': 5}
index_list = ['sub_' + str(i + 1) for i in range(24)]

class_0_list = []
class_1_list = []
class_2_list = []
class_3_list = []
class_4_list = []
class_5_list = []
for i in index_list:
    X, y = load_data(i)
    class_0 = y.count(0)
    class_0_list.append(class_0)
    class_1 = y.count(1)
    class_1_list.append(class_1)
    class_2 = y.count(2)
    class_2_list.append(class_2)
    class_3 = y.count(3)
    class_3_list.append(class_3)
    class_4 = y.count(4)
    class_4_list.append(class_4)
    class_5 = y.count(5)
    class_5_list.append(class_5)

print()
import matplotlib.pyplot as plt
import numpy as np

data = [class_0_list, class_1_list, class_2_list, class_3_list, class_4_list, class_5_list]

fig = plt.figure()

# Creating axes instance
#ax = fig.add_axes([0, 0, 1, 1])
# x-axis labels
#ax.set_yticklabels(['class_0', 'class_1', 'class_2', 'class_3', 'class_4'])
# Creating plot
#bp = ax.boxplot(data)
plt.boxplot(data)
plt.xlabel("classes",fontsize=13)
plt.ylabel("number of samples",fontsize=13)
# show plot
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.savefig('HAR_classes_distribution.png')
print()


