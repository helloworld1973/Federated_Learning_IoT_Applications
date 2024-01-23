import os
import warnings
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
from glob import glob
from gtda.time_series import SlidingWindow

num = '{{user_name}}'
Batch_Size = 10
Test_Size = 0.4
Segment_Size = 128

activity_codes = {'dws': 0, 'jog': 1, 'sit': 2, 'std': 3, 'ups': 4, 'wlk': 5}
activity_types = list(activity_codes.keys())



def load_data(num):
    # Load All data:
    Folders = glob('../A_DeviceMotion_data/*_*')
    Folders = [s for s in Folders if "csv" not in s]

    X = []
    y = []
    for j in Folders:
        Csv = glob(j + '/'+num+'.csv')[0]
        label = int(activity_codes[j[23:26]])
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test_Size, stratify=y, random_state=1234)

    return X_train, X_test, y_train, y_test


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


X_train, X_test, y_train, y_test = load_data(num=num)

trainloader = DataLoader(CustomDataset(X_train, y_train), batch_size=Batch_Size, shuffle=True, drop_last=True)
testloader = DataLoader(CustomDataset(X_test, y_test))
num_examples = {"trainset" : len(X_train), "testset" : len(X_test)}
# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cpu")


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


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    for _ in range(epochs):
        for signals, labels in tqdm(trainloader):
            signal = signals.view(Batch_Size, 12, 128)
            optimizer.zero_grad()
            criterion(net(signal.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
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


net = Net().to(DEVICE).double()

'''
list_of_files = [fname for fname in glob("../model_round_*")]
latest_round_file = max(list_of_files, key=os.path.getctime)
print("Loading pre-trained model from: ", latest_round_file)
state_dict = torch.load(latest_round_file)
net.load_state_dict(state_dict)
'''

# Define Flower client
class PatientClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


# Start client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=PatientClient())