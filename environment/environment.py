from environment.env_var import Box, OneD
from environment.dataset import Uav, Dataset

import numpy as np
import math
from enum import Enum
        
class Env:
    def __init__(self, noUav, end, timestep = 86400, baseStationCoords=(0,0)):
        self.noUav = noUav
        self.timestep = timestep
        
        self.baseStationCoords = baseStationCoords
        self.start = 0
        self.end = end
        
    def reset(self):
        self.done = False
        self.throughput = []

    
    def step(self):
        pass
    
    def render(self):
        pass