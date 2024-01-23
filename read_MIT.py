import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt

if os.path.isdir("mitdb"):
    print('You already have the data.')
else:
    wfdb.dl_database('mitdb', 'mitdb')


def beats_types():
    normal_beats = ['N', 'L', 'R', 'e', 'j']
    sup_beats = ['A', 'a', 'J', 'S']
    ven_beats = ['V', 'E']
    fusion_beats = ['F']
    unknown_beat = ['/', 'f', 'Q']
    return normal_beats, sup_beats, ven_beats, fusion_beats, unknown_beat


classes = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
# classes = {'Normal beat', 'Supreventricular Ectopic Beat', 'Ventricular Ectopic beat', 'Fusion Beat', 'Unknown Beat'}

age_list = []
gender_list = []
index_list = [i for i in range(100, 103)] + [i for i in range(104, 110)] + [i for i in range(111, 120)] \
                + [i for i in range(121, 125)] + \
                [i for i in range(200, 204)] + [i for i in range(205, 206)] + [i for i in range(207, 211)] + \
                [i for i in range(212, 216)] + [i for i in range(217, 218)] + [i for i in range(220, 224)] + \
                [i for i in range(228, 229)] + [i for i in range(230, 235)]



class_0_list = []
class_1_list = []
class_2_list = []
class_3_list = []
class_4_list = []
for i in index_list:
    record = wfdb.rdsamp('mitdb/' + str(i))
    record_comments = record[1]['comments'][0].split(' ')
    age_list.append(int(record_comments[0]))
    gender_list.append(record_comments[1])
    annotation_symbols = wfdb.rdann('mitdb/' + str(i), 'atr').symbol
    print()
    for i, a_label in enumerate(annotation_symbols):
        Beat_types = beats_types()
        if a_label in Beat_types[0]:
            annotation_symbols[i] = 0
        elif a_label in Beat_types[1]:
            annotation_symbols[i] = 1
        elif a_label in Beat_types[2]:
            annotation_symbols[i] = 2
        elif a_label in Beat_types[3]:
            annotation_symbols[i] = 3
        elif a_label in Beat_types[4]:
            annotation_symbols[i] = 4
        else:
            #print('label has some not includes')
            annotation_symbols[i] = 4

    class_0 = annotation_symbols.count(0)
    class_0_list.append(class_0)
    class_1 = annotation_symbols.count(1)
    class_1_list.append(class_1)
    class_2 = annotation_symbols.count(2)
    class_2_list.append(class_2)
    class_3 = annotation_symbols.count(3)
    class_3_list.append(class_3)
    class_4 = annotation_symbols.count(4)
    class_4_list.append(class_4)

import matplotlib.pyplot as plt
import numpy as np

data = [class_0_list, class_1_list, class_2_list, class_3_list, class_4_list]

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
plt.savefig('MIT_classes_distribution.png')
print()
