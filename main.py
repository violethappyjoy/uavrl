import numpy as np
from environment.environment import Uav, Env
from environment.dataset import Dataset

# env = Env(10, windowSize=2, end=10)
env = Env(10)
current = env.reset()
# print(current)
# print(np.array(current).reshape(-1, *current.shape))
# print(env.calcReward(env.actionSpace.sample()))
print(env.observationSpace.shape)
# while True:
#     action = env.actionSpace.sample()
#     state, reward, done = env.step(action)
#     print(done, reward, env.uavId)
#     # print(reward)
#     # print(env.uavId)
#     if done:
#         # print(state)
#         # print(env.uavId)
#         for _, uav in enumerate(env.cluster):
#             snir = uav.calcSNIR()
#             throughtput = uav.calcThroughput()
#             print([uav.tx, snir, throughtput])
#         break

# print(env.cluster[0].calcThroughput())
# # print(4e+7)

# dataset = Dataset(3,10)
# dataset.genDataset()
# print(dataset.data)
# print(dataset.data.shape)
