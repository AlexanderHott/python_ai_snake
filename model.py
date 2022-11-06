import os
import time

import torch
import torch.nn.functional as F
from torch import nn, optim


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, filename=None):
        if filename is None:
            filename = f"model_{time.time()}.pth"

        if not os.path.isdir("./models"):
            os.mkdir("./models")

        torch.save(self.state_dict(), f"./models/{filename}")


class QTrainer:
    def __init__(self, model, learning_rate, gamma) -> None:
        self.learning_rate = learning_rate
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

            done = (game_over,)

            pred = self.model(state)

            target = pred.clone()
            for i in range(len(done)):
                Q_new = reward[i]
                if not done[i]:
                    Q_new = reward[i] + self.gamma * torch.max(
                        self.model(next_state[i])
                    )

                target[i][torch.argmax(action).item()] = Q_new

            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            self.optimizer.step()
