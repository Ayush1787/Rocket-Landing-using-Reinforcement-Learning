import torch
import torch.nn as nn
import torch.nn.functional as F
from Rocket import ACTIONS

class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()

        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 256)

        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, len(ACTIONS))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value(x)
        advantage = self.advantage(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values