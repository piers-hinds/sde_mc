import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, final_activation=None):
        assert len(layer_sizes) > 0, "At least one hidden layer required."
        super(Mlp, self).__init__()
        self.num_layers = len(layer_sizes)

        layers = [nn.BatchNorm1d(input_size), nn.Linear(input_size, layer_sizes[0]), nn.BatchNorm1d(layer_sizes[0]),
                  nn.Tanh()]
        for i in range(self.num_layers-1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.BatchNorm1d(layer_sizes[i+1]), nn.Tanh()]
        layers += [nn.Linear(layer_sizes[self.num_layers-1], output_size)]
        if final_activation is not None:
            layers += [final_activation]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PathData(Dataset):
    def __init__(self, data):
        super(PathData, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :2], self.data[idx, 2:]


class PosELU(nn.Module):
    def __init__(self):
        super(PosELU, self).__init__()

    def forward(self, x):
        return F.elu(x, 1., False) + 1.0


class TanhExp(nn.Module):
    def __init__(self):
        super(TanhExp, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.exp(x))