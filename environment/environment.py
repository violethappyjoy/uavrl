from environment.env_var import Box, OneD
from environment.dataset import Uav, Dataset

import numpy as np
import math
from enum import Enum
from collections import deque
        
class Actions(Enum):
    ignore = 0
    choose = 1  

class Env:
    def __init__(self, noUav, windowSize = 1800, end = 86400, baseStationCoords=(np.random.uniform(0, 100), np.random.uniform(0, 100))):
        self.noUav = noUav
        # self.timestep = timestep
        self.windowSize = windowSize
        
        self.shape = (windowSize, 1)
        self.actionSpace = OneD(len(Actions))
        
        self.baseStationCoords = baseStationCoords
        self.start = 0
        self.end = end
        self.state = []
        self.memoryT = deque(maxlen = self.windowSize)
        self.memoryP = deque(maxlen = self.windowSize)
        
    def reset(self):
        self.done = False
        self.current = self.start
        self.reward = 0
        self.uavId = 0
        
        # self.snir = []
        # self.throughput = []
        return self.getState()

    def getState(self):
        self.cluster = [
            Uav(
                v=np.random.uniform(150, 190),
                tx=np.random.uniform(21, 27) + self.reward if self.uavId == id else np.random.uniform(21, 27),
                noise=174,
                baseStationCoords=self.baseStationCoords
            ) for id in range(self.noUav)
        ]
        for _, uav in enumerate(self.cluster):
            snir = uav.calcSNIR()
            throughtput = uav.calcThroughput()
            self.state.append([uav.tx, snir, throughtput])
            
        print(self.state)
        
        return np.array(self.state)
    
    def calcReward(self, action):
        if action == Actions.ignore.value:
            return 0
        elif action == Actions.choose.value:
            stateArr = np.array(self.state)
            self.uavId = np.argmax(stateArr[:, 1])
            return self.uavId
    
    def step(self, action):
        pass
    
    def render(self):
        pass