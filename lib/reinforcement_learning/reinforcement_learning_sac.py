# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:15:49 2023

@author: awei
reinforcement_learning_sac
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import pandas as pd

from __init__ import path
from base import base_connect_database

# Define the actor and critic neural networks using PyTorch
class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        #action = torch.tanh(self.fc3(x))  # [-1, 1]
        action = torch.sigmoid(self.fc3(x))  # [0, 1]
        return action

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACCritic, self).__init__()
        self.fc_state = nn.Linear(state_dim, 64)
        self.fc_action = nn.Linear(action_dim, 64)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state, action):
        x_state = torch.relu(self.fc_state(state))
        x_action = torch.relu(self.fc_action(action))
        
        # Ensure the dimensions are compatible for concatenation
        if len(x_state.shape) == 1:
            x_state = x_state.unsqueeze(0)
        if len(x_action.shape) == 1:
            x_action = x_action.unsqueeze(0)

        x = torch.cat([x_state, x_action], dim=1)
        x = torch.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = SACActor(state_dim, action_dim)
        self.critic = SACCritic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state)
        return action.detach().numpy()

    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state)

        # Critic loss
        q_value = self.critic(state, action)
        next_action = self.actor(next_state)
        next_q_value = self.critic(next_state, next_action.detach())
        target_q = reward + 0.99 * next_q_value * (1 - done)
        critic_loss = nn.MSELoss()(q_value, target_q)

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def simulate_trading(self, data):
        # Initial
        capital_init = 1000_000
        capital = capital_init
        position = 0
        state = data.iloc[0].values
        #total_reward = 0

        for i in range(1, len(data)):
            # bug % in capital
            # sell % in position
            
            action = self.get_action(state)
            buy_ratio, buy_price, sell_ratio, sell_price = action
            buy_price *= 2 # Because sigmoid reduces the confidence interval to [0,1] and enlarges it to [0,2]
            sell_price *= 2
            
            close_price = data['close'].iloc[i]
            #reward = (close_price - data['close'].iloc[i - 1]) * position  # Profit or loss from holding position
            #total_reward += reward
            total_reward = position * close_price + capital - capital_init

            # Buy/Sell decision based on SAC agent's action
            if buy > 0:
                buy
                #position += capital // close_price  # Buy as many shares as possible
                #capital -= position * close_price
            if sell > 0 and position > 0:
                volume = position * sell
                capital += volume * close_price 
                position -= sell*position

            next_state = data.iloc[i].values
            self.train(state, action, total_reward, next_state, False)  # Training on each step
            state = next_state
        return total_reward, total_reward + capital_init
    
    
if __name__ == '__main__':
    # Simulated data (replace this with your actual stock data)
# =============================================================================
#     np.random.seed(0)
#     dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='B')
#     data = pd.DataFrame(index=dates)
#     data['close'] = np.random.randn(len(dates)).cumsum() + 100
# =============================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-03-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-12-01', help='进行回测的结束时间')
    args = parser.parse_args()

    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        sql = f"SELECT * FROM prediction_stock_price_test WHERE date >= '{args.date_start}' AND date < '{args.date_end}' and code='sz.002230' "
        backtest_df = pd.read_sql(sql, con=conn.engine)
    
    print(backtest_df)
    
    backtest_input_df = backtest_df[['rearLowPctChgPred', 'rearHighPctChgPred', 'rearDiffPctChgPred', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']]
    
    # Initialize agent and environment
    state_dim = len(backtest_input_df.columns)
    action_dim = 4
    agent = SACAgent(state_dim, action_dim)
    
    # Simulate trading for 100 episodes
    for episode in range(100):
        total_reward, final_balance = agent.simulate_trading(backtest_input_df)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Final Balance: {final_balance}")

#self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)