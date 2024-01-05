import numpy as np
from environment.environment import Uav, Env
from environment.dataset import Dataset

env = Env(10)
current = env.reset()
print(current)
print(np.array(current).reshape(-1, *current.shape))
# print(env.cluster[0].calcThroughput())
# # print(4e+7)

# dataset = Dataset(3,10)
# dataset.genDataset()
# print(dataset.data)
# print(dataset.data.shape)
