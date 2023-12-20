# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:15:49 2023

@author: awei
reinforcement_learning_sac
"""
import argparse
import math

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
        action_mean = torch.sigmoid(self.fc3(x))  # [0, 1]
        log_std = torch.zeros_like(action_mean)  # Fixed log_std for simplicity

        return action_mean, log_std

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

    def get_action(self, state, exploration_noise=0.1):
        with torch.no_grad():
            state = torch.FloatTensor(state)  # Convert state to PyTorch tensor
            action_mean, log_std = self.actor(state)
            std = torch.exp(log_std)
            raw_action = action_mean + std * torch.randn_like(std)
            #action = torch.tanh(raw_action)  # Scale to [-1, 1]
            action = torch.sigmoid(raw_action)
            
            # Add exploration noise
            action += exploration_noise * torch.randn_like(action)
            action = torch.clamp(action, 0.0, 1.0)  # Clip to [0, 1]

            return action.detach().numpy()
        
    def train(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state)

        # Critic loss
        q_value = self.critic(state, action)
        next_action, _ = self.actor(next_state)
        next_q_value = self.critic(next_state, next_action.detach())
        target_q = reward + 0.99 * next_q_value * (1 - done)
        critic_loss = nn.MSELoss()(q_value, target_q)

        # Actor loss
        action_mean, _ = self.actor(state)
        actor_loss = -self.critic(state, action_mean).mean()

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
        total_volume = 0
        #volume = 0
        state = data.iloc[0].values
        #total_reward = 0

        for i in range(1, len(data)):
            #print(f'{i}: capital: {capital:.2f}')
            # sell_ratio % in valume
            # buy_ratio % in capital
            
            action = self.get_action(state)
            #print(action)
            buy_ratio, buy_price_ratio, sell_ratio, sell_price_ratio = action  # action_dim = 4
            
            # Because sigmoid reduces the confidence interval to [0,1] and enlarges it to [0,2]
            buy_price_ratio *= 2
            sell_price_ratio *= 2
            #print(buy_ratio, buy_price_ratio, sell_ratio, sell_price_ratio)
            preclose = data['close'].iloc[i - 1]
            rearLowPctChgPred, rearHighPctChgPred, high, low, close, preclose, date = backtest_df.loc[i, ['rearLowPctChgPred', 'rearHighPctChgPred', 'high', 'low', 'close', 'preclose', 'date']]
            
            total_reward = total_volume * close + capital - capital_init

            # Buy/Sell decision based on SAC agent's action
            # pending order
            sell_price = (1 + rearHighPctChgPred * sell_price_ratio) * preclose
            if (sell_ratio > 0) and (total_volume > 0): # The selling price is less than or equal to the highest price
                # trade
                if sell_price <= high:
                    volume = total_volume if total_volume<100 else math.floor(sell_ratio * total_volume)  # Less than 100 shares can only be sold in full
                    capital += (volume * sell_price)
                    total_volume -= volume
                    print(f'sell_date: {date} |total_volume:{total_volume} |volume: {volume} |sell_price: {sell_price:.2f} |high: {high}')
                
            # pending order
            buy_price = (1 + rearLowPctChgPred * buy_price_ratio) * preclose
            if (buy_ratio > 0) and (buy_ratio * capital >= buy_price * 100):  # Need to buy at least 100 shares, 
                # trade
                if buy_price >= low:
                    volume = math.floor((buy_ratio * capital) / buy_price)
                    total_volume += volume
                    capital -= (volume * buy_price)
                    print(f'buy_date: {date} |total_volume:{total_volume} |volume: {volume} |buy_price: {buy_price:.2f} |low: {low}')
                    
            next_state = data.iloc[i].values
            self.train(state, action, total_reward, next_state, False)  # Training on each step
            state = next_state
        print(f'capital: {capital:.2f} |total_volume: {total_volume} |close: {close}')
        return total_reward, capital + total_volume * close  # total_reward + capital_init
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-03-01', help='Start time for backtesting')
    parser.add_argument('--date_end', type=str, default='2023-12-01', help='End time for backtesting')
    args = parser.parse_args()

    print(f'Start time for backtesting: {args.date_start}\nEnd time for backtesting: {args.date_end}')
    
    with base_connect_database.engine_conn('postgre') as conn:
        sql = f"SELECT * FROM prediction_stock_price_test WHERE date >= '{args.date_start}' AND date < '{args.date_end}' and code='sz.002230' "
        backtest_df = pd.read_sql(sql, con=conn.engine)
    
    print(backtest_df)
    
    backtest_input_df = backtest_df[['rearLowPctChgPred', 'rearHighPctChgPred', 'rearDiffPctChgPred', 'open', 'high',
                                     'low', 'close', 'volume', 'amount', 'turn', 'pctChg']]
    
    # Initialize agent and environment
    state_dim = len(backtest_input_df.columns)
    action_dim = 4
    agent = SACAgent(state_dim, action_dim)
    
    # Simulate trading for 100 episodes
    for episode in range(50):
        print('===========================')
        total_reward, final_balance = agent.simulate_trading(backtest_input_df)
        print(f"Episode: {episode + 1} |Total Reward: {total_reward:.2f} |Final Balance: {final_balance:.2f}")

#self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
