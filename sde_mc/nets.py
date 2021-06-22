import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Mlp(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, activation=nn.Tanh, final_activation=None):
        assert len(layer_sizes) > 0, "At least one hidden layer required."
        super(Mlp, self).__init__()
        self.num_layers = len(layer_sizes)

        layers = [nn.BatchNorm1d(input_size), nn.Linear(input_size, layer_sizes[0]), nn.BatchNorm1d(layer_sizes[0]),
                  activation()]
        for i in range(self.num_layers-1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.BatchNorm1d(layer_sizes[i+1]), activation()]
        layers += [nn.Linear(layer_sizes[self.num_layers-1], output_size)]
        if final_activation is not None:
            layers += [final_activation()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PathData(Dataset):
    def __init__(self, data, dim=1):
        super(PathData, self).__init__()
        self.data = data
        self.dim = dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :(1+self.dim)], self.data[idx, (1+self.dim  ):]


class NormalPathData(Dataset):
    def __init__(self, paths, payoffs, normals):
        self.paths = paths[:, :-1]
        self.payoffs = payoffs
        self.normals = normals

    def __len__(self):
        return len(self.payoffs)

    def __getitem__(self, idx):
        return (self.paths[idx], self.normals[idx]), self.payoffs[idx]


class NormalJumpsPathData(Dataset):
    def __init__(self, paths, left_paths, payoffs, normals, jumps):
        self.paths = paths[:, :-1]
        self.left_paths = left_paths[:, :-1]
        self.payoffs = payoffs
        self.normals = normals
        self.jumps = jumps

    def __len__(self):
        return len(self.payoffs)

    def __getitem__(self, idx):
        return (self.paths[idx], self.left_paths[idx], self.normals[idx], self.jumps[idx]), self.payoffs[idx]

