import numpy as np
from environment.environment import Uav, Env

print((np.random.uniform(0, 100), np.random.uniform(0, 100))[1])
env = Env(1, baseStationCoords=(np.random.uniform(0, 100), np.random.uniform(0, 100)))
print(env.cluster[0].calcThroughput())
# print(4e+7)