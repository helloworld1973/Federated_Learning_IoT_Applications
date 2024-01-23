import warnings
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from gtda.time_series import SlidingWindow
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.optim as optim
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cpu")

class CNN_individual():
    def __init__(self, user_num):
        self.num = user_num
        self.Test_Size = 0.4
        self.Batch_Size = 10

    def beats_types(self):
        # classes = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
        normal_beats = ['N', 'L', 'R', 'e', 'j']
        sup_beats = ['A', 'a', 'J', 'S']
        ven_beats = ['V', 'E']
        fusion_beats = ['F']
        unknown_beat = ['/', 'f', 'Q']
        return normal_beats, sup_beats, ven_beats, fusion_beats, unknown_beat

    def load_data(self):
        # Load All data:
        Folders = glob('./A_DeviceMotion_data/*_*')
        Folders = [s for s in Folders if "csv" not in s]
        activity_codes = {'dws': 0, 'jog': 1, 'sit': 2, 'std': 3, 'ups': 4, 'wlk': 5}
        activity_types = list(activity_codes.keys())
        Segment_Size = 128
        Test_Size = 0.4

        X = []
        y = []
        for j in Folders:
            Csv = glob(j + '/' + self.num + '.csv')[0]
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

            sliding_bag = SlidingWindow(size=Segment_Size, stride=int(Segment_Size / 2))
            X_bags = sliding_bag.fit_transform(x_a_activity_a_user_complete_data)
            y_bags = [label for i in range(len(X_bags))]

            for a_X_bag in X_bags:
                X.append(a_X_bag)
            for a_y_bag in y_bags:
                y.append(a_y_bag)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test_Size, stratify=y, random_state=1234)

        return X_train, X_test, y_train, y_test

    def train(self, net, trainloader, epochs, testloader):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
        acc_list = []
        for _ in range(epochs):
            for signals, labels in tqdm(trainloader):
                signal = signals.view(self.Batch_Size, 12, 128)
                optimizer.zero_grad()
                criterion(net(signal.to(DEVICE)), labels.to(DEVICE)).backward()
                optimizer.step()
            loss, acc = self.test(net, testloader)
            acc_list.append(acc)
        return acc_list

    def test(self, net, testloader):
        """Validate the model on the test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for signals, labels in tqdm(testloader):
                signal = signals.view(12, 128)
                outputs = net(signal.to(DEVICE))
                labels = labels.to(DEVICE)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        return loss / len(testloader.dataset), correct / total

    def run(self, rounds, X_train, X_test, y_train, y_test):
        # X_train, X_test, y_train, y_test = self.load_data(num=self.num)
        trainloader = DataLoader(CustomDataset(X_train, y_train), batch_size=self.Batch_Size, shuffle=True, drop_last=True)
        testloader = DataLoader(CustomDataset(X_test, y_test))
        net = Net().to(DEVICE).double()
        acc_list = self.train(net, trainloader, rounds, testloader)
        return acc_list



class CustomDataset(Dataset):
    def __init__(self, data_features, data_label):
        self.data_features = data_features
        self.data_label = data_label

    def __len__(self):
        return len(self.data_features)

    def __getitem__(self, index):
        data = self.data_features[index]
        labels = self.data_label[index]
        return data, labels




# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(12, 3, 15)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(3, 6, 15)

        # self.fc1 = nn.Linear(6 * 29, 20)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(6 * 21, 6)

        '''
        self.fc1 = nn.Linear(16 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
        '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 21)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc3(x)





