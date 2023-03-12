import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# Set up Alpaca API key and secret
API_KEY = 'ALPACA_API_KEY' #Paper Trading
API_SECRET = 'ALPACA_SECRET_KEY' #Paper Trading
api = tradeapi.REST(API_KEY, API_SECRET, api_version='v2')

# Load historical data for Apple stock
end_dt = datetime.today().strftime('%Y-%m-%d')
start_dt = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
data = api.get_barset('AAPL', 'day', start=start_dt, end=end_dt).df['AAPL']


# Preprocess data
new_data = pd.DataFrame(index=range(0,len(data)),columns=['Date', 'Close', 'High', 'Low', 'Volume'])
for i in range(0,len(data)):
    new_data['Date'][i] = data.index[i]
    new_data['Close'][i] = data['close'][i]
    new_data['High'][i] = data['high'][i]
    new_data['Low'][i] = data['low'][i]
    new_data['Volume'][i] = data['volume'][i]
new_data = new_data.set_index('Date')


# Define our trading environment
class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.balance = initial_balance
        self.shares = 0
        self.profit = 0

        # Define our action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,))

    def step(self, action):
        # Execute a trade based on the action
        if action > 0 and self.balance > 0:
            shares = min(action * self.balance / self.data['Close'][self.current_step], self.balance / self.data['Close'][self.current_step])
            self.balance -= shares * self.data['Close'][self.current_step]
            self.shares += shares
        elif action < 0 and self.shares > 0:
            shares = min(-action * self.shares, self.shares)
            self.balance += shares * self.data['Close'][self.current_step]
            self.shares -= shares

        # Move to the next time step

        self.current_step += 1

        # Calculate the reward
        self.profit = self.balance + self.shares * self.data['Close'][self.current_step]
        reward = self.profit - self.initial_balance

        # Check if the episode is over
        done = self.current_step >= len(self.data) - 1

        # Calculate the observation
        obs = np.array([
            self.data['Close'][self.current_step] / self.data['Close'].max(),
            self.data['High'][self.current_step] / self.data['High'].max(),
            self.data['Low'][self.current_step] / self.data['Low'].max(),
            self.data['Volume'][self.current_step] / self.data['Volume'].max(),
            self.shares / (self.balance + self.shares * self.data['Close'][self.current_step])
        ])

        return obs, reward, done, {}
        
    def reset(self):
        # Reset the environment to its initial state
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.profit = 0
        return np.array([
            self.data['Close'][self.current_step] / self.data['Close'].max(),
            self.data['High'][self.current_step] / self.data['High'].max(),
            self.data['Low'][self.current_step] / self.data['Low'].max(),
            self.data['Volume'][self.current_step] / self.data['Volume'].max(),
            self.shares / (self.balance + self.shares * self.data['Close'][self.current_step])
        ])

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Define our trading agent
def create_agent(env):
    model = Sequential()
    model.add(Dense(64, input_shape=(5,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer=Adam())

    agent = PPO2('MlpPolicy', env, verbose=0)
    agent.learn(total_timesteps=20000)
    return agent

# Train the agent
env = TradingEnv(new_data)
env = DummyVecEnv([lambda: env])
agent = create_agent(env)


# Use the trained model to generate trading signals
def generate_signal(model, data):
    obs = np.array([
        data['close'] / data['close'].max(),
        data['high'] / data['high'].max(),
        data['low'] / data['low'].max(),
        data['volume'] / data['volume'].max(),
        0
    ])
    action, _ = model.predict(obs)
    return action[0]
