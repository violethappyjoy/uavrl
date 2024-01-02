from environment.env_var import Box, OneD

import numpy as np
import math
from enum import Enum

class Uav:
    def __init__(self, v, tx, rx, noise, baseStationCoords):
        self.v = v # Velocity
        self.tx = tx # Transmission Power
        self.rx = rx # Receive power
        self.noise = noise # Extra Noise (eg. Thermal)
        self.intref = 2*tx # Induce Noise
        self.base = baseStationCoords 
        self.coords = (np.random.uniform(0, 100), np.random.uniform(0, 100))
        self.d = self.calcDist()
        self.chGain = 1/self.d # Channel Gain
        
        if self.d <= 26.5:
            self.B = 4e+7 # 40Mhz
        elif self.d>26.5 and self.d<= 57.3:
            self.B = 2e+7 # 20Mhz
        elif self.d>57.3 and self.d<= 116:
            self.B = 1e+7 # 10 Mhz
        else:
            self.B = 5e+6 # 5 Mhz
        # self.d = self.calcDist()
        # self.sinr = self.chGain/(self.intref + self.noise)
        
    def calcDist(self):
        x1, y1 = self.base
        x2, y2 = self.coords 
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def calcSNIR(self):
        numerator = self.tx * (self.chGain)**2
        dinom = self.noise + self.intref
        return numerator/dinom
    
    def calcThroughput(self):
        return self.B * (np.log2(1 + self.calcSNIR()))
        
class Env:
    def __init__(self, noUav, end, baseStationCoords=(0,0)):
        self.noUav = noUav
        self.baseStationCoords = baseStationCoords
        self.start = 0
        self.end = end
        
    def reset(self):
        self.done = False
        self.throughput = []
        self.cluster = [Uav(
            v=np.random.uniform(150, 190),
            tx=np.random.uniform(21, 27),
            rx=np.random.uniform(75, 96),
            noise=174,
            baseStationCoords=self.baseStationCoords
        )for _ in range(self.noUav)]

    
    def step(self):
        pass
    
    def render(self):
        pass