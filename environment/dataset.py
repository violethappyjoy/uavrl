import numpy as np
import math

class Uav:
    def __init__(self, v, tx, noise, baseStationCoords):
        self.v = v * (5/18) # Velocity Kmph
        # self.tx = tx * self.v # Transmission Power dBm
        self.tx = self.calcTx(tx) # in watts
        self.noise = noise # Extra Noise dB
        self.intref = np.random.rand()*tx # Induce Noise
        self.base = baseStationCoords 
        self.coords = (np.random.uniform(22, 250), np.random.uniform(22, 250))
        self.d = self.calcDist()
        self.chGain = (1/self.d) * self.v # Channel Gain
        
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
    # def calcTx(self, dBm):
    #     power = (dBm - 30)/10
    #     tx = (10**power) * self.v
    #     return tx
    def calcTx(self, dBm):
        try:
            power = (dBm - 30)/10
            tx = (10**power) * self.v
            return tx
        except OverflowError as e:
            print(f"OverFlowError: {e}")
            print(f"Value causing the overflow - power: {power}, dBm = {dBm}")
            power = 18/10
            tx = ((10**power)/1000) * self.v  # Convert dB to watts
            return tx

    
    def calcDist(self):
        x1, y1 = self.base
        x2, y2 = self.coords 
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def calcSNIR(self):
        numerator = self.tx * (self.chGain)**2
        dinom = self.noise + self.intref
        return numerator/dinom  
    
    # def calcThroughput(self):
    #     snir = self.calcSNIR()
    #     return self.B * (np.log2(1 + snir))
    
    def calcThroughput(self):
        try:
            snir = self.calcSNIR()
            return self.B * (np.log2(1 + snir))
        except RuntimeWarning as warning:
            print(f"Warning: {warning}")
            print(f"Values causing the warning - snir: {snir}, B: {self.B}")
            return np.nan
    
class Dataset:
    def __init__(self, noUav, timestep):
            self.noUav = noUav
            self.timestep = timestep
            self.baseStationCoords = (np.random.uniform(0, 272), np.random.uniform(0, 272))
            self.data = np.zeros((timestep,2*self.noUav))
            
    def genDataset(self, idx, reward):
        transx = np.random.uniform(21, 27) + reward
        if transx >= 27:
            transx = 27
        for t in range(self.timestep):
            self.cluster = [Uav(
            v=np.random.uniform(150, 190), #
            tx= transx if idx == id else np.random.uniform(21, 27),
            noise=174,
            baseStationCoords=self.baseStationCoords
        ) for id in range(self.noUav)]
            for i, uav in enumerate(self.cluster):
                snir = uav.calcSNIR()
                throughput = uav.calcThroughput()
                self.data[t, 2 * i] = snir
                self.data[t, 2 * i + 1] = throughput