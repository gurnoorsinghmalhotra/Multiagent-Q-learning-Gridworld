"""
dqn.py  -  DQN model for GridWorld
This module defines the DQN class, which implements a simple feedforward neural network
to approximate the Q-values for the GridWorld environment
"""

import copy
import torch

class DQN:
    def __init__(self, state_size=7, action_size=4):
        l1 = state_size
        l2 = 24
        l3 = 24
        l4 = action_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4)
        )

        self.model2 = copy.deepcopy(self.model)
        self.model2.load_state_dict(self.model.state_dict())
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def update_target(self):
        self.model2.load_state_dict(self.model.state_dict())

    def get_qvals(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        return self.model(state)

    def get_maxQ(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.model2(state)
        return torch.max(q_values).item()

    def train_one_step(self, states, actions, targets):
        state1_batch = torch.cat([torch.from_numpy(s).float().unsqueeze(0) for s in states], dim=0)
        action_batch = torch.tensor(actions)
        Q1 = self.model(state1_batch)

        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(1)).squeeze()
        Y = torch.tensor(targets).float()
        loss = self.loss_fn(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, filename="checkpoint.pt"):
        """Save online-network weights only."""
        torch.save(self.model.state_dict(), filename)

    def load(self, filename="checkpoint.pt"):
        """Load weights into both online and target networks."""
        self.model.load_state_dict(torch.load(filename))
        self.update_target()