import numpy as np
import pandas as pd

# Define the environment
class Environment:
  def __init__(self, data):
    self.data = data
    self.state = 0
    self.end_state = len(data) - 1
    self.n_actions = 3

  def take_action(self, action, current_money):
    price = self.data.iloc[self.state]['Close']
    if action == 0:
      return current_money, 0
    elif action == 1:
      return current_money - price, -price
    else:
      return current_money + price, price
  
  def get_state(self):
    return self.data.iloc[self.state][['Open', 'Close', 'Low', 'High']]

# Define the agent
class Agent:
  def __init__(self, n_actions, n_features):
    self.n_actions = n_actions
    self.n_features = n_fsseatures
    self.Q = np.zeros((n_actions, n_features))

  def choose_action(self, state):
    if np.random.uniform(0, 1) < 0.1:
      action = np.random.choice(self.n_actions)
    else:
      action = np.argmax(self.Q[:, state])
    return action
  
  def update_Q(self, state, action, reward, next_state):
    alpha = 0.1
    gamma = 0.9
    max_q = max(self.Q[:, next_state])
    self.Q[action, state] = (1-alpha)*self.Q[action, state] + alpha*(reward + gamma*max_q)

# Train the agent
def train(agent, env, n_episodes):
  for episode in range(n_episodes):
    done = False
    state = env.state
    current_money = 100
    while not done:
      action = agent.choose_action(state)
      current_money, reward = env.take_action(action, current_money)
      next_state = env.state + 1
      agent.update_Q(state, action, reward, next_state)
      state = next_state
      done = env.state == env.end_state

# Play the game
def play(agent, env, initial_money):
  done = False
  state = env.state
  current_money = initial_money
  while not done:
    action = agent.choose_action(state)
    current_money, reward = env.take_action(action, current_money)
    state = env.state + 1
    done = env.state == env.end_state
  print("Game Over! Final money:", current_money)

# Main function
if __name__ == "__main__":
  data = pd.read_csv('stock_data.csv')
  env = Environment(data)
  agent = Agent(env.n_actions, env.n_features)
  train(agent, env, 1000)
  play(agent, env
