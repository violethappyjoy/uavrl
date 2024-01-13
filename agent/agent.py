# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.optimizers import Adam
# import numpy as np
# import random
# from collections import deque

# class Agent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=50000)  # Experience replay buffer
#         self.gamma = 0.99  # Discount factor
#         self.epsilon = 1.0  # Exploration-exploitation trade-off
#         self.epsilon_decay = 0.995
#         self.epsilon_min = 0.01
#         self.learning_rate = 0.001
#         self.model = self.build_model()

#     def build_model(self):
#         with tf.device("/GPU:0"):
#             model = Sequential()
#             model.add(Flatten(input_shape=(self.state_size,)))
#             model.add(Dense(22, activation='relu'))
#             model.add(Dense(15, activation='relu'))
#             model.add(Dense(self.action_size, activation='linear'))
#             model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
#             return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.choice(range(self.action_size))
        
#         # Reshape state to match the expected input shape
#         state = np.reshape(state, [1, self.state_size])
        
#         q_values = self.model.predict(state)
#         return np.argmax(q_values[0])

#     def replay(self, batch_size):
#         if len(self.memory) < batch_size:
#             return
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     def train(self, state, action, reward, next_state, done):
#         state = np.reshape(state, [1, self.state_size])
#         next_state = np.reshape(next_state, [1, self.state_size])
#         self.remember(state, action, reward, next_state, done)
#         self.replay(batch_size=32)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
from tqdm import tqdm  # Import tqdm for the progress bar

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # Experience replay buffer
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.episode_rewards = []

    def build_model(self):
        with tf.device("/GPU:0"):
            model = Sequential()
            model.add(Flatten(input_shape=(self.state_size,)))
            model.add(Dense(22, activation='relu'))
            model.add(Dense(15, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        # Reshape state to match the expected input shape
        state = np.reshape(state, [1, self.state_size])
        
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        episode_losses = []  # Initialize episode losses list
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            episode_losses.append(loss)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return episode_losses

    def train(self, state, action, reward, next_state, done, num_episodes=None):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        self.remember(state, action, reward, next_state, done)

        if done:
            # Episode is finished, replay and collect episode statistics
            episode_losses = self.replay(batch_size=32)
            episode_reward = np.sum([item[2] for item in list(self.memory)])  # Sum of all rewards in the episode
            self.episode_rewards.append(episode_reward)

            # Print episode information
            tqdm.write(f"Episode: {len(self.episode_rewards)}, Epsilon: {self.epsilon}, Episode Reward: {episode_reward}, Mean Loss: {np.mean(episode_losses)}")

            # Reset memory for the next episode
            self.memory.clear()

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        if num_episodes is not None and len(self.episode_rewards) >= num_episodes:
            return  # Stop training after reaching the specified number of episodes
