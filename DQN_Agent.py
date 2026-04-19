import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from DQN import DuelingDQN
from Rocket import (
    ALPHA, MEMORY_SIZE, EPSILON,
    ACTIONS, BATCH_SIZE, GAMMA,
    EPSILON_MIN, EPSILON_DECAY
)

class DQNAgent:

    def __init__(self):

        self.tau = 0.005

        self.model = DuelingDQN()
        self.target_model = DuelingDQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON

        self.episode_rewards = []
        self.losses = []
        self.success_count = 0
        self.total_episodes = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(ACTIONS)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):

        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN
        next_actions = self.model(next_states).argmax(1)
        next_q = self.target_model(next_states).gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1).detach()

        expected_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, expected_q)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        # Soft Update
        for target_param, param in zip(
            self.target_model.parameters(),
            self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data +
                (1 - self.tau) * target_param.data
            )

    def save_model(self, path="rocket_model.pth"):
        torch.save(self.model.state_dict(), path)