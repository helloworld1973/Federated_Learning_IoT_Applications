from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Union
import torch.nn.functional as F
import flwr as fl
import numpy as np
import torch
from flwr.common import Metrics, FitRes, Parameters, Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from torch import nn
import joblib

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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 21)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc3(x)


'''
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 29)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return self.fc3(x)
'''

net = Net().to(DEVICE).double()


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            joblib.dump(aggregated_ndarrays, f"model_round_{server_round}.pth")
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            #params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            #state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            #net.load_state_dict(state_dict, strict=True)

            # Save the model
            #torch.save(net.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Aggregate evaluation accuracy using weighted average."""

            if not results:
                return None, {}

            # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            # Aggregate and print custom metric
            aggregated_accuracy = sum(accuracies) / sum(examples)
            print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

            # Return aggregated loss and metrics (i.e., aggregated accuracy)
            return aggregated_loss, {"accuracy": aggregated_accuracy}

# Create strategy and run server
strategy = SaveModelStrategy(min_available_clients=46) #fraction_fit=1.0, min_available_clients=2, min_evaluate_clients=13
fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=100), strategy=strategy)
