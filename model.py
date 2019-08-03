import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_sizes=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list of int): Number of nodes in each hidden layer
        """
        assert(len(hidden_sizes) > 0, "hidden_sizes parameter needs to be a list of at least one integer")
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_first = nn.Linear(state_size, hidden_sizes[0])

        self.fc_list = nn.ModuleList()
        if (len(hidden_sizes) > 1):
            self.fc_list.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(0,len(hidden_sizes)-1)])

        self.fc_last = nn.Linear(hidden_sizes[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc_first(state))

        for fc_hidden in self.fc_list:
            x = F.relu(fc_hidden(x))

        return self.fc_last(x)

