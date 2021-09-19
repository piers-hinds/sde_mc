import torch.nn as nn
from torch.utils.data import Dataset


class Mlp(nn.Module):
    """Multilayer perceptron (MLP)"""

    def __init__(self, input_size, layer_sizes, output_size, activation=nn.ReLU, final_activation=None,
                 batch_norm=True, batch_norm_init=True):
        """
        :param input_size: int
            Input dimensions

        :param layer_sizes: list of ints
            The sizes of the hidden layers

        :param output_size: int
            Output dimension

        :param activation: nn.Module (default = nn.ReLU)
            The activation function used after each linear layer

        :param final_activation: nn.Module (default = None)
            The final activation function

        :param batch_norm: bool (default = True)
            If True, uses a batch norma layer before each activation

        :param batch_norm_init: bool (default = True)
            It True, uses a batch norm layer on the inputs
        """
        assert len(layer_sizes) > 0, "At least one hidden layer required."
        super(Mlp, self).__init__()
        self.num_layers = len(layer_sizes)
        layers = []

        if batch_norm_init:
            layers += [nn.BatchNorm1d(input_size)]
        layers += [nn.Linear(input_size, layer_sizes[0])]
        if batch_norm:
            layers += [nn.BatchNorm1d(layer_sizes[0])]
        layers += [activation()]

        for i in range(self.num_layers-1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1])]
            if batch_norm:
                layers += [nn.BatchNorm1d(layer_sizes[i+1])]
            layers += [activation()]
        layers += [nn.Linear(layer_sizes[self.num_layers-1], output_size)]
        if final_activation is not None:
            layers += [final_activation()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
    def __init__(self, paths, payoffs, normals, jumps):
        self.paths = paths[:, :-1]
        self.payoffs = payoffs
        self.normals = normals
        self.jumps = jumps

    def __len__(self):
        return len(self.payoffs)

    def __getitem__(self, idx):
        return (self.paths[idx], self.normals[idx], self.jumps[idx]), self.payoffs[idx]


class AdaptedPathData(Dataset):
    def __init__(self, paths, payoffs, normals, jumps, jump_times, left_paths, time_paths, jump_paths, total_steps):
        self.paths = paths[:, :-1]
        self.payoffs = payoffs
        self.normals = normals
        self.jumps = jumps
        self.jump_times = jump_times
        self.left_paths = left_paths[:, :-1]
        self.time_paths = time_paths[:, :-1]
        self.jump_paths = jump_paths[:, :-1]
        self.total_steps = total_steps

    def __len__(self):
        return len(self.payoffs)

    def __getitem__(self, idx):
        return (self.paths[idx], self.normals[idx], self.jumps[idx], self.jump_times[idx], self.left_paths[idx],
                self.time_paths[idx], self.jump_paths[idx]), self.payoffs[idx]
