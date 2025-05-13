import numpy as np


class Agent:
    def __init__(self, learning_rate, gamma, number_of_actions, number_of_states, min_eps, max_eps, eps_decrease_rate):
        self.lr = learning_rate
        self.gamma = gamma
        self.noa = number_of_actions
        self.nos = number_of_states
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.eps_dr = eps_decrease_rate

        self.Q = {}
        self.init_Q()

    def init_Q(self):
        for s in range(self.nos):
            for a in range(self.noa):
                self.Q[s, a] = 0.0

    def choose_action(self, state):
        if np.random.rand() <= self.max_eps:
            action = np.random.choice(np.array([a for a in range(self.noa)]))
        else:
            expected_rewards = np.array([self.Q[state, a] for a in range(self.noa)])
            action = np.argmax(expected_rewards)
        return action

    def decrease_epsilon(self):
        if self.max_eps > self.min_eps:
            self.max_eps *= self.eps_dr
        else:
            self.max_eps = self.min_eps

    def learn(self, state, action, resulted_state, reward):
        expected_rewards = np.array([self.Q[resulted_state, a] for a in range(self.noa)])
        best_action = np.argmax(expected_rewards)
        self.Q[state, action] += self.lr * (reward + self.gamma * self.Q[resulted_state, best_action] - self.Q[state, action])
        self.decrease_epsilon()