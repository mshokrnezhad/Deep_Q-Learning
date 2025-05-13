#v2
import gym
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('FrozenLake-v0')
agent = Agent(learning_rate=0.001, gamma=0.9, number_of_actions=4, number_of_states=16,
              min_eps=0.01, max_eps=1, eps_decrease_rate=0.9999995)
numberOfGames = 500000
scores = []
win_pcts = []

for i in range(numberOfGames):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        agent.learn(obs, action, obs_, reward)
        score += reward
        obs = obs_
    scores.append(score)
    if i % 100 == 0:
        win_pct = np.mean(scores[-100:])
        win_pcts.append(win_pct)
        if i % 1000 == 0:
            print("i:", i, "win_pct:", win_pct, "eps:", agent.max_eps)

plt.plot(win_pcts)
plt.show()

# v1
# import gym
#
# env = gym.make('FrozenLake-v0')
# numberOfGames = 1000
# score = 0
# policy = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}
#
# for i in range(numberOfGames):
#     obs = env.reset()
#     done = False
#     while not done:
#         #action = env.action_space.sample()
#         action = policy[obs]
#         obs, reward, done, info = env.step(action)
#         if reward == 1:
#             score += 1
# print("success rate: " + str(score/numberOfGames))
