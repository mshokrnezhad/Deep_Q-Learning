import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class LinearDQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDQN, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # use GPU if available
        self.to(self.device)  # move whole model to device

    def forward(self, state):  # forward propagation includes defining layers
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions


class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(n_actions)]
        self.Q = LinearDQN(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)  # converting observation to tensor
            q_values = self.Q.forward(state)
            action = T.argmax(q_values).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

    def learn(self, state, action, reward, resulted_state):
        processed_state = T.tensor(state, dtype=T.float).to(self.Q.device)
        processed_action = T.tensor(action).to(self.Q.device)
        processed_reward = T.tensor(reward).to(self.Q.device)
        processed_resulted_state = T.tensor(resulted_state, dtype=T.float).to(self.Q.device)

        prediction = self.Q.forward(processed_state)[processed_action]
        next_result = self.Q.forward(processed_resulted_state).max()
        target = processed_reward + self.gamma * next_result

        loss = self.Q.loss(target, prediction).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()
