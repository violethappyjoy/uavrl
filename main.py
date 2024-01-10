import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
import os
import time

from environment.environment import Uav, Env
from environment.dataset import Dataset
from agent.agent import Agent

# env = Env(noUav=5, windowSize=10, end=100)
# # env = Env(10)
# current = env.reset()
# # print(current)
# # print(np.array(current).reshape(-1, *current.shape))
# # print(env.calcReward(env.actionSpace.sample()))
# print(env.observationSpace.shape)
# for episode in range(3):
#     while True:
#         action = env.actionSpace.sample()
#         state, reward, done = env.step(action)
#         # print(done, reward, env.uavId)
#         # print(reward)
#         # print(env.uavId)
#         if done:
#             # print(state)
#             # print(env.uavId)
#             for _, uav in enumerate(env.cluster):
#                 snir = uav.calcSNIR()
#                 throughtput = uav.calcThroughput()
#                 print([uav.tx, snir, throughtput])
#             break
#     env.render(episode)
    # env.plotRewards(episode)
    # env.plotThroughput(episode)

# print(env.cluster[0].calcThroughput())
# # print(4e+7)

# dataset = Dataset(3,10)
# dataset.genDataset()
# print(dataset.data)
# print(dataset.data.shape)

env = Env(noUav=5, windowSize=10, end=100)
EPISODES = 10
# MAX_TIMESTEPS = 

# Create an instance of the DQNAgent
state_size = env.observationSize
action_size = 2
agent = Agent(state_size, action_size)
print(state_size)

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.train(state, action, reward, next_state, done)
        print(done)
        state = next_state
        if done:
            break
        
    env.render(episode)