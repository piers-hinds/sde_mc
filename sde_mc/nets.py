import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class ControlVariate(nn.Module):
    """Abstract class for control variates"""

    def __init__(self, sequential, device):
        """
        :param sequential: bool
            It True, data will be passed in sequence

        :param device: str
            The device to store the model on
        """
        super(ControlVariate, self).__init__()
        self.sequential = sequential
        self.device = device


class ZeroFunction(ControlVariate):
    """A control variate that maps any input to zero"""

    def __init__(self, output_dim):
        """
        :param output_dim: int
            The dimension of the output. The shape of the output will be (bs, output_dim)
        """
        super(ZeroFunction, self).__init__(sequential=False, device=None)
        self.output_dim = output_dim

    def forward(self, x):
        bs = x.shape[0]
        return torch.zeros((bs, self.output_dim), device=x.device)


class Mlp(ControlVariate):
    """Multilayer perceptron (MLP)"""

    def __init__(self, input_size, layer_sizes, output_size, activation=nn.ReLU, final_activation=None,
                 batch_norm=True, batch_norm_init=True, device='cpu'):
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

        :param device: str (default = 'cpu')
            The device to store the model on
        """
        assert len(layer_sizes) > 0, "At least one hidden layer required."
        super(Mlp, self).__init__(sequential=False, device=device)
        self.num_layers = len(layer_sizes)
        layers = []

        if batch_norm_init:
            layers += [nn.BatchNorm1d(input_size, device=device)]
        layers += [nn.Linear(input_size, layer_sizes[0], device=device)]
        if batch_norm:
            layers += [nn.BatchNorm1d(layer_sizes[0], device=device)]
        layers += [activation()]

        for i in range(self.num_layers - 1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1], device=device)]
            if batch_norm:
                layers += [nn.BatchNorm1d(layer_sizes[i + 1], device=device)]
            layers += [activation()]
        layers += [nn.Linear(layer_sizes[self.num_layers - 1], output_size, device=device)]
        if final_activation is not None:
            layers += [final_activation()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Lstm(ControlVariate):
    """Long short-term memory RNN (LSTM) model"""

    def __init__(self, in_dim, hidden_dim, out_dim, device='cpu'):
        """
        :param in_dim: int
            The dimension of the input data

        :param hidden_dim: int
            The dimension of the hidden state

        :param out_dim: int
            The dimension of the output

        :param device: str (default = 'cpu')
            The device to store the model on
        """
        super(Lstm, self).__init__(sequential=True, device=device)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True).to(device)
        self.lin = nn.Linear(hidden_dim, out_dim).to(device)

    def forward(self, x):
        h = self.init_hidden(x.shape[0])
        out, h = self.lstm(x, h)
        out = self.lin(out)
        return out

    def init_hidden(self, bs):
        return torch.zeros((1, bs, self.hidden_dim), device=self.device), torch.zeros((1, bs, self.hidden_dim),
                                                                                      device=self.device)


class Gru(ControlVariate):
    """Gated Recurrent Unit (GRU)"""

    def __init__(self, in_dim, hidden_dim, out_dim, device='cpu'):
        """
        :param in_dim: int
            The dimension of the input data

        :param hidden_dim: int
            The dimension of the hidden state

        :param out_dim: int
            The dimension of the output

        :param device: str (default = 'cpu')
            The device to store the model on
        """
        super(Gru, self).__init__(sequential=True, device=device)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True).to(device)
        self.lin = nn.Linear(hidden_dim, out_dim).to(device)

    def forward(self, x):
        h = self.init_hidden(x.shape[0])
        out, h = self.gru(x, h)
        out = self.lin(out)
        return out

    def init_hidden(self, bs):
        return torch.zeros((1, bs, self.hidden_dim), device=self.device)


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
    def __init__(self, paths, payoffs, normals, left_paths, time_paths, jump_paths, total_steps):
        self.paths = paths[:, :-1]
        self.payoffs = payoffs
        self.normals = normals
        self.left_paths = left_paths[:, :-1]
        self.time_paths = time_paths[:, :-1]
        self.jump_paths = jump_paths[:, :-1]
        self.total_steps = total_steps

    def __len__(self):
        return len(self.payoffs)

    def __getitem__(self, idx):
        return (self.paths[idx], self.normals[idx], self.left_paths[idx],
                self.time_paths[idx], self.jump_paths[idx]), self.payoffs[idx]


def get_mlps(problem, num_layers, hidden_size, device):
    d = problem.dim() + 1
    brown_dim = problem.solver.sde.brown_dim
    layers = [hidden_size + d for _ in range(num_layers)]
    f = Mlp(d, layers, brown_dim, batch_norm=False, device=device)
    if problem.solver.has_jumps:
        g = Mlp(d, layers, d - 1, batch_norm=False, device=device)
        return [f, g]
    else:
        return f


def get_opt(models):
    if isinstance(models, list):
        opt = optim.Adam(list(models[0].parameters()) + list(models[1].parameters()))
        return opt
    else:
        opt = optim.Adam(models.parameters())
        return opt
