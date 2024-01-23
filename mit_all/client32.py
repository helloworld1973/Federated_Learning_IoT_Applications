import warnings
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset

num = 213
Batch_Size = 10
Test_Size = 0.4

def beats_types():
    # classes = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
    normal_beats = ['N', 'L', 'R', 'e', 'j']
    sup_beats = ['A', 'a', 'J', 'S']
    ven_beats = ['V', 'E']
    fusion_beats = ['F']
    unknown_beat = ['/', 'f', 'Q']
    return normal_beats, sup_beats, ven_beats, fusion_beats, unknown_beat


def load_data(num):
    record = wfdb.rdsamp('../mitdb/' + str(num))
    annotation = wfdb.rdann('../mitdb/' + str(num), 'atr')

    record_data = record[0][:, 0]  # MLII
    annotation_index = annotation.sample
    annotation_symbols = annotation.symbol

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

    X = []
    y = []
    Length_RRI = len(annotation_index)
    for L in range(Length_RRI - 2):
        Ind1 = int((annotation_index[L] + annotation_index[L + 1]) / 2)
        Ind2 = int((annotation_index[L + 1] + annotation_index[L + 2]) / 2)

        Symb = annotation_symbols[L + 1]
        y.append(Symb)
        Sign = record_data[Ind1:Ind2]
        Resamp = signal.resample(x=Sign, num=128)
        X.append(Resamp)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test_Size)

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
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(3, 6, 5)

        #self.fc1 = nn.Linear(6 * 29, 20)
        #self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(6 * 29, 5)

        '''
        self.fc1 = nn.Linear(16 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
        '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 29)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    for _ in range(epochs):
        for signals, labels in tqdm(trainloader):
            signal = signals.view(Batch_Size, 1, 128)
            optimizer.zero_grad()
            criterion(net(signal.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for signals, labels in tqdm(testloader):
            signal = signals.view(1, 128)
            outputs = net(signal.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


net = Net().to(DEVICE).double()


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