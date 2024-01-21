import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
from tqdm import tqdm  # Import tqdm for the progress bar

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os

# tf.get_logger().setLevel()
tf.keras.utils.disable_interactive_logging()
class Agent:
    def __init__(self, state_size, action_size, replay_buffer_size=50000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=replay_buffer_size)  # Experience replay buffer
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.target_model = self.build_model()
        self.model = self.build_model()
        self.episode_rewards = []
        self.episode_mean_loss = []
        self.q_values_history = []

    def build_model(self):
        with tf.device("/GPU:0"):
            model = Sequential()
            model.add(Flatten(input_shape=(self.state_size,)))
            model.add(Dense(22, activation='relu'))
            model.add(Dense(15, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.learning_rate, clipvalue=1.0))
            return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
        # tf.keras.utils.disable_interactive_logging()
        episode_losses = []  # Initialize episode losses list
        if len(self.memory) < batch_size:
            return episode_losses

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            q_value_before_update = target_f[0][action]
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=2)
            loss = history.history['loss'][0]
            episode_losses.append(loss)
            
            q_value_after_update = self.model.predict(state)[0][action]
            self.q_values_history.append((q_value_before_update, q_value_after_update))

        return episode_losses

    def train(self, state, action, reward, next_state, done, num_episodes=None):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        self.remember(state, action, reward, next_state, done)

        if done:
            # Episode is finished, replay and collect episode statistics
            episode_losses = self.replay(batch_size=64)
            episode_reward = np.sum([item[2] for item in list(self.memory)])  # Sum of all rewards in the episode
            self.episode_rewards.append(episode_reward)
            self.episode_mean_loss.append(np.mean(episode_losses))

            # Print episode information
            tqdm.write(f"Episode: {len(self.episode_rewards)}, Epsilon: {self.epsilon}, Episode Reward: {episode_reward}, Mean Loss: {np.mean(episode_losses)}")
            self.render()
            # Reset memory for the next episode
            self.memory.clear()
            

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        if num_episodes is not None and len(self.episode_rewards) >= num_episodes:
            return  # Stop training after reaching the specified number of episodes

    def render(self):
        timestep = range(len(self.episode_rewards))
        plt.plot(timestep, self.episode_rewards)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        os.makedirs('aggRewad', exist_ok=True)
        plt.savefig(f'aggRewad/rewards_aggregate.png')
        plt.close()
        
        timestep = range(len(self.episode_mean_loss))
        plt.plot(timestep, self.episode_mean_loss)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        os.makedirs('aggLoss', exist_ok=True)
        plt.savefig(f'aggLoss/loss_aggregate.png')
        plt.close()
        
        self.plot_q_value_convergence()
        
    def plot_q_value_convergence(self):
        timestep = range(len(self.q_values_history))
        q_values_before_update = [item[0] for item in self.q_values_history]
        q_values_after_update = [item[1] for item in self.q_values_history]

        plt.plot(timestep, q_values_before_update, label='Q-value Before Update')
        plt.plot(timestep, q_values_after_update, label='Q-value After Update')
        plt.xlabel('Step')
        plt.ylabel('Q-value')
        plt.legend()
        os.makedirs('qValueConvergence', exist_ok=True)
        plt.savefig(f'qValueConvergence/q_value_convergence.png')
        plt.close()