# import numpy as np
# import random
# import tensorflow as tf
# from tqdm import tqdm
import os
# import time

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from environment.environment import Uav, Env
# from environment.dataset import Dataset
from agent.agent import Agent
from tqdm import tqdm

# env = Env(noUav=5, windowSize=10, end=100)
env = Env(5)
# current = env.reset()
# action = env.actionSpace.sample()
# print(action)
# state, reward, done = env.step(action)
# print(state, reward, done)

# print(env.actionSpace.sample())
# print(current)
# # print(np.array(current).reshape(-1, *current.shape))
# # print(env.calcReward(env.actionSpace.sample()))
# print(env.observationSpace.shape)
# while True:
#     action = env.actionSpace.sample()
#     state, reward, done = env.step(action)
#     # print(done, reward, env.uavId)
#     # print(reward)
#     print(env.uavId, reward)
#     if done:
#         # print(state)
#         # print(env.uavId)
#         for _, uav in enumerate(env.cluster):
#             snir = uav.calcSNIR()
#             throughtput = uav.calcThroughput()
#             print([uav.tx, snir, throughtput])
#         break
# env.render('test')

# print(env.cluster[0].calcThroughput())
# # print(4e+7)

# dataset = Dataset(3,10)
# dataset.genDataset()
# print(dataset.data)
# print(dataset.data.shape)

# env = Env(noUav=5, windowSize=10, end=100)
EPISODES = 100
# MAX_TIMESTEPS = 

# Create an instance of the DQNAgent
state_size = env.observationSize
print(state_size)
action_size = env.actionSpace.n
print(action_size)
agent = Agent(state_size, action_size)
# print(state_size)

# Training loop
for episode in tqdm(range(EPISODES), desc='Training Episodes'):
    state = env.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.train(state, action, reward, next_state, done)
        # print(done)
        state = next_state
        if done:
            break
        
    if episode%10 == 0 or episode == 0:
        env.render(episode)