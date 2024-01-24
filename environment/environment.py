# from environment.env_var import Box, OneD
# from environment.dataset import Uav, Dataset

# import numpy as np
# import math
# from enum import Enum
# import os
# from collections import deque
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
        
# class Actions(Enum):
#     ignore = 0
#     choose = 1  

# class Env:
#     def __init__(self, noUav, windowSize = 1800, end = 86400, baseStationCoords=(np.random.uniform(0, 100), np.random.uniform(0, 100))):
#         self.noUav = noUav
#         # self.timestep = timestep
#         self.windowSize = windowSize
        
#         self.shape = (windowSize, 1)
#         self.actionSpace = OneD(len(Actions))
#         self.observationSize = self.noUav * 3
#         self.observationSpace = Box(low=np.full(self.observationSize, -np.inf), high=np.full(self.observationSize, np.inf), size=(self.observationSize, ))
        
        
#         self.baseStationCoords = baseStationCoords
#         self.start = 0
#         self.end = end
#         self.state = deque(maxlen = self.noUav)
#         self.memoryT = deque([0] * self.windowSize, maxlen=self.windowSize)
#         # self.memoryP = deque([0] * self.windowSize, maxlen=self.windowSize)
        
        
        
#     def reset(self):
#         self.done = False
#         self.current = self.start
#         self.reward = 0
#         self.uavId = 0
        
#         self.throughput = []
#         self.totalReward = []
#         # self.snir = []
#         # self.throughput = []
#         return self.getState()

#     def getState(self):
#         self.cluster = [
#             Uav(
#                 v=np.random.uniform(80, 100),
#                 tx=np.random.uniform(21, 27) + self.reward/2 if self.uavId == id else np.random.uniform(21, 27),
#                 noise=174,
#                 baseStationCoords=self.baseStationCoords
#             ) for id in range(self.noUav)
#         ]
#         for _, uav in enumerate(self.cluster):
#             snir = uav.calcSNIR()
#             throughtput = uav.calcThroughput()
#             self.state.append([uav.tx, snir, throughtput])
            
#         # print(self.state)
        
#         return np.array(self.state)
    
#     def calcReward(self, action):
#         if action == Actions.ignore.value:
#             return 0
#         elif action == Actions.choose.value:
#             stateArr = np.array(self.state)
#             # choose max SNIR
#             self.uavId = np.argmax(stateArr[:, 1])
#             # return self.uavId, self.cluster[self.uavId].calcSNIR(), self.cluster[self.uavId].calcThroughput()
#             # print(self.uavId)
#             throughput = self.cluster[self.uavId].calcThroughput()
#             # snir = self.cluster[self.uavId].calcSNIR()
#             if throughput >= self.memoryT[-1] and throughput >= max(self.memoryT):
#                 return self.cluster[self.uavId].tx/12
#             elif throughput >= self.memoryT[-1] and throughput < max(self.memoryT):
#                 return self.cluster[self.uavId].tx/24
#             else:
#                 return -self.cluster[self.uavId].tx/12
                
            
    
#     def step(self, action): 
#         self.memoryT.append(self.cluster[self.uavId].calcThroughput())
#         self.current+=1
        
#         if self.current == self.end:
#             self.done = True
            
#         self.reward = self.calcReward(action)
#         if self.current%(2*self.windowSize) or self.current==1:
#             self.totalReward.append(self.reward)
#             th = [item[2] for item in self.state]
#             self.throughput.append(th)
#             # print(self.throughput)
        
#         step = self.getState()
        
#         return step, self.reward, self.done
    
#     # def plotThroughput(self):
#     #     uavThroughputs = list(map(list, zip(*self.throughput)))
#     #     timestep = range(len(self.throughput))
#     #     cmap = get_cmap('viridis')
        
#     #     fig, ax = plt.subplots(figsize=(10, 5)) 
        
#     #     for uavIdx, uavData in enumerate(uavThroughputs):
#     #         color = cmap(uavIdx / len(uavThroughputs))
#     #         ax.plot(timestep, uavData, label=f'UAV {uavIdx + 1}', color=color)
            
#     #     ax.set_xlabel('Timestep')
#     #     ax.set_ylabel('Throughput')
#     #     ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     #     plt.savefig('throughputGraphs/throughput.png')
    
#     def plotThroughput(self, episode):
#         uavThroughputs = list(map(list, zip(*self.throughput)))
#         timestep = range(len(self.throughput))
#         cmap = get_cmap('viridis')

#         # Check if the directory exists, create if not
#         os.makedirs('throughputGraphs', exist_ok=True)

#         # Counter variable for plot index
#         plot_index = 1

#         for uavIdx, uavData in enumerate(uavThroughputs):
#             color = cmap(uavIdx / len(uavThroughputs))
#             fig, ax = plt.subplots(figsize=(10, 5))

#             ax.plot(timestep, uavData, label=f'UAV {uavIdx + 1}', color=color)
#             ax.set_xlabel('Timestep')
#             ax.set_ylabel('Throughput')
#             ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
#             # Save the plot to the 'throughputGraphs/' directory with a unique index
#             plt.savefig(f'throughputGraphs/throughput_{plot_index}_episode_{episode}.png')

#             # Increment the counter
#             plot_index += 1
#             plt.close()
    
#     def plotRewards(self, episode):
#         timestep = range(len(self.totalReward))
#         plt.plot(timestep, self.totalReward)
#         plt.xlabel('Timestep')
#         plt.ylabel('Throughput')
#         os.makedirs('rewardGraphs', exist_ok=True)

#     # Save the plot to the 'rewardGraphs/' directory
#         plt.savefig(f'rewardGraphs/rewards_episode_{episode}.png')
#         plt.close()
    
#     def render(self, episode):
#         self.plotRewards(episode)
#         self.plotThroughput(episode)
        
from environment.dataset import Uav   
from environment.env_var import Box, OneD

import numpy as np
import math
# from enum import Enum
import os
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# class Actions(Enum):
#     pass

class Env:
    def __init__(self, noUav, windowSize = 1800, end = 86400, baseStationCoords=(np.random.uniform(0, 272), np.random.uniform(0, 272))):
        self.noUav = noUav
        # self.timestep = timestep
        self.windowSize = windowSize
        # self.createActionsEnum()
        
        self.shape = (windowSize, 1)
        self.actionSpace = OneD(self.noUav) 
        self.observationSize = self.noUav * 4
        self.observationSpace = Box(low=np.full(self.observationSize, -np.inf), high=np.full(self.observationSize, np.inf), size=(self.observationSize, ))
        
        
        self.baseStationCoords = baseStationCoords
        self.start = 0
        self.end = end
        self.state = deque(maxlen = self.noUav)
        self.memoryT = deque([0] * self.windowSize, maxlen=self.windowSize)
        # self.memoryP = deque([0] * self.windowSize, maxlen=self.windowSize)
        
    # def createActionsEnum(self):
    #     self.ActionsEnum = Enum("ActionsEnum", {f"uav{i}": i for i in range(self.noUav)}, type=Actions)
        # for i in range(self.noUav):
        #     enum_num = f"uav{i}"
        #     Actions[enum_num] = i  
        
    def reset(self):
        self.done = False
        self.current = self.start
        self.reward = 0
        self.uavId = 0
        self.maxUavId = 0
        
        self.throughput = []
        self.totalReward = []
        # self.snir = []
        # self.throughput = []
        return self.getState()

    def normalize_observation(self, observation):
        # Normalize observation values to a common scale
        normalized_observation = (observation - observation.mean()) / (observation.std() + 1e-8)
        return normalized_observation
    
    def getState(self):
        self.cluster = []
        for id in range(self.noUav):
            if self.current == self.start:
                velo = np.random.uniform(150,190)
                txDbm = np.random.uniform(21, 27)
                # print(txDbm)
                # coords = (np.random.uniform(22, 250), np.random.uniform(22, 250))
            elif self.uavId == id:
                v = self.state[self.uavId][1]
                if self.reward>0:
                    # print("TEST")
                    velo = v
                # elif v < 145 and self.reward>0:
                #     velo = np.random.uniform(100, v)
                else:
                    velo = np.random.uniform(150,190)
                txDbm = np.random.uniform(21, 27) + self.reward
            else:
                velo = np.random.uniform(150,190)
                txDbm = np.random.uniform(21, 27) - self.reward/2
                
            if txDbm >= self.noUav * 27:
                txDbm = self. noUav * 27
            if txDbm < 21 / self.noUav:
                txDbm = 21 / self.noUav
            obj = Uav(
                v = velo,
                tx = txDbm,
                noise = 174,
                baseStationCoords = self.baseStationCoords
            )
            self.cluster.append(obj)
            
        # self.cluster = [
        #     Uav(
        #         v=np.random.uniform(100, 190),
        #         tx=np.random.uniform(21, 27) + self.reward if self.uavId == id else np.random.uniform(21, 27),
        #         noise=174,
        #         baseStationCoords=self.baseStationCoords
        #     ) for id in range(self.noUav)
        # ]
        for _, uav in enumerate(self.cluster):
            snir = uav.calcSNIR()
            throughtput = uav.calcThroughput()
            self.state.append([uav.dBmtx, uav.v, snir, throughtput])
            
        # print(self.state)
        
        return self.normalize_observation(np.array(self.state))
    
    def calcReward(self, action):
        stateArr = np.array(self.state)
        self.maxUavId = np.argmax(stateArr[:, 2])
        self.uavId = action
        throughputSelected = self.cluster[self.uavId].calcThroughput()
        snirSelected = self.cluster[self.uavId].calcSNIR()
        snirMax = self.cluster[self.maxUavId].calcSNIR()
        if snirSelected>=snirMax and throughputSelected >= max(self.memoryT):
            reward = self.cluster[self.uavId].dBmtx
            # print("test3")
            return reward 
        elif snirSelected>=snirMax and throughputSelected >= self.memoryT[-1]:
            reward = self.cluster[self.uavId].dBmtx
            # print("test2")
            return reward / 2
        elif snirSelected<snirMax:
            reward = self.cluster[self.maxUavId].dBmtx
            # print("test1")
            return -(reward / 2)
        else:
            reward = self.cluster[self.maxUavId].dBmtx
            # print("test")
            return 0
        
        
        # throughput = self.cluster[action].calcThroughput()
        # if throughput >= self.memoryT[-1] and throughput >= max(self.memoryT):
        #     return self.cluster[self.uavId].tx/6
        # elif throughput >= self.memoryT[-1] and throughput < max(self.memoryT):
        #     return self.cluster[self.uavId].tx/12
        # else:
        #     return -self.cluster[self.uavId].tx/6 
    
    def step(self, action): 
        self.current+=1
        self.memoryT.append(self.cluster[self.uavId].calcThroughput())
        
        if self.current == self.end:
            self.done = True
            
        self.reward = self.calcReward(action)
        if self.current%(2*self.windowSize) == 0 or self.current==1:
            self.totalReward.append(self.reward)
            th = [item[2] for item in self.state]
            self.throughput.append(th)
            # print(self.throughput)
        
        step = self.getState()
        
        return step, self.reward, self.done
    
    def plotThroughputAgg(self, episode):
        uavThroughputs = list(map(list, zip(*self.throughput)))
        timestep = range(len(self.throughput))
        cmap = get_cmap('viridis')
        
        fig, ax = plt.subplots() 
        
        for uavIdx, uavData in enumerate(uavThroughputs):
            color = cmap(uavIdx / len(uavThroughputs))
            ax.plot(timestep, uavData, label=f'UAV {uavIdx + 1}', color=color)
            
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Throughput')
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        os.makedirs('throughputGraphsAgg', exist_ok=True)
        plt.savefig(f'throughputGraphsAgg/throughput_{episode}.png', dpi=300)
        plt.close()
    
    def plotThroughput(self, episode):
        uavThroughputs = list(map(list, zip(*self.throughput)))
        timestep = range(len(self.throughput))
        cmap = get_cmap('viridis')

        # Check if the directory exists, create if not
        os.makedirs('throughputGraphs', exist_ok=True)

        # Counter variable for plot index
        plot_index = 1

        for uavIdx, uavData in enumerate(uavThroughputs):
            color = cmap(uavIdx / len(uavThroughputs))
            fig, ax = plt.subplots()

            ax.plot(timestep, uavData, label=f'UAV {uavIdx + 1}', color=color)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Throughput')
            # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Save the plot to the 'throughputGraphs/' directory with a unique index
            plt.savefig(f'throughputGraphs/throughput_{plot_index}_episode_{episode}.png', dpi=300)

            # Increment the counter
            plot_index += 1
            plt.close()
    
    def plotRewards(self, episode):
        timestep = range(len(self.totalReward))
        plt.plot(timestep, self.totalReward)
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        os.makedirs('rewardGraphs', exist_ok=True)

    # Save the plot to the 'rewardGraphs/' directory
        plt.savefig(f'rewardGraphs/rewards_episode_{episode}.png', dpi=300)
        plt.close()
    
    def render(self, episode):
        self.plotRewards(episode)
        self.plotThroughput(episode)
        self.plotThroughputAgg(episode)