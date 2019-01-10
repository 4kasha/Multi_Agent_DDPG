import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, use_bn, use_reset):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): size of hidden_layers
            use_bn (bool): use batch norm or not.
            use_reset (bool): weights initialization used in original paper
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        if use_bn:
            self.batchnorm_input = nn.BatchNorm1d(state_size)
            self.batchnorm_layers = nn.ModuleList([nn.BatchNorm1d(hidden_layers[0])])
            self.batchnorm_layers.extend([nn.BatchNorm1d(layer) for layer in hidden_layers[1:]])
        
        use_bias = not use_bn
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0], bias=use_bias)])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2, bias=use_bias) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)
        
        self.use_bn = use_bn
        
        if use_reset:
            self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hidden_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        self.output.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.use_bn:
            state = self.batchnorm_input(state)
            for linear, batch in zip(self.hidden_layers, self.batchnorm_layers):
                state = F.relu(batch(linear(state)))
        else:
            for linear in self.hidden_layers:
                state = F.relu(linear(state))
            
        return F.tanh(self.output(state))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, use_bn, use_reset):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): size of hidden_layers
            use_bn (bool): use batch norm or not.
            use_reset (bool): weights initialization used in original paper
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        if use_bn:
            self.bn0 = nn.BatchNorm1d(state_size)
            self.bn1 = nn.BatchNorm1d(hidden_layers[0])

        use_bias = not use_bn
        self.fcs1 = nn.Linear(state_size, hidden_layers[0], bias=use_bias)
        self.fcs2 = nn.Linear(hidden_layers[0]+action_size, hidden_layers[1])
        self.fcs3 = nn.Linear(hidden_layers[1], 1)

        self.use_bn = use_bn
        
        if use_reset:
            self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fcs2.weight.data.uniform_(*hidden_init(self.fcs2))
        self.fcs3.weight.data.uniform_(-3e-3, 3e-3)
        self.fcs3.bias.data.uniform_(-3e-3, 3e-3)        

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.use_bn:
            state = self.bn0(state)
            xs = F.relu(self.bn1(self.fcs1(state)))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fcs2(x))
        else:
            xs = F.relu(self.fcs1(state))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fcs2(x))

        return self.fcs3(x)