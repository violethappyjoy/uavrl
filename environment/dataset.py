import numpy as np
import math

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
    
class Dataset:
    def __init__(self, noUav, timestep):
            self.noUav = noUav
            self.timestep = timestep
            self.baseStationCoords = (np.random.uniform(0, 100), np.random.uniform(0, 100))
            self.data = np.zeros((timestep,2*self.noUav))
            
    def genDataset(self):
        for t in range(self.timestep):
            self.cluster = [Uav(
            v=np.random.uniform(150, 190),
            tx=np.random.uniform(21, 27),
            rx=np.random.uniform(75, 96),
            noise=174,
            baseStationCoords=self.baseStationCoords
        ) for _ in range(self.noUav)]
            for i, uav in enumerate(self.cluster):
                snir = uav.calcSNIR()
                throughput = uav.calcThroughput()
                self.data[t, 2 * i] = snir
                self.data[t, 2 * i + 1] = throughput