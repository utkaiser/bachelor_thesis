import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import logging

HMAX_NORMALIZE = 1
INITIAL_ACCOUNT_BALANCE=100000
STOCK_DIM = 2
TRANSACTION_FEE_PERCENT = 0.01
REWARD_SCALING = 1e-4

class StockEnvTrain(gym.Env):
    def __init__(self, df, asset_name, day=0):
        self.asset_name = asset_name
        self.day = day
        self.df = df
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (15,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.gold.values.tolist() + \
                     self.data.interest.values.tolist() + \
                     self.data.index.values.tolist() + \
                     self.data.similar.values.tolist() + \
                     self.data.vix.values.tolist()
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        self._seed()
        self.stop = False
        self.last_improvement_index = 0
        self.require_improvement = 10000000 #15???????????
        self.highest_value = 0
        self.min_delta = 0
        self.epoch = 0

    def _sell_stock(self, index, action):

        # perform sell action based on the sign of the action
        if self.state[index+STOCK_DIM+1] > 0 and index+0 == 0:
            #update balance
            self.state[0] += self.state[index+1]*1 * (1- TRANSACTION_FEE_PERCENT)
            self.state[index+STOCK_DIM+1] -= 1
            self.cost +=self.state[index+1]*1 * \
             TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:
            pass

    
    def _buy_stock(self, index, action):

        available_amount = self.state[0] // self.state[index+1]
        #update balance
        if index+1 == 1 and available_amount>1:
            self.state[0] -= self.state[index+1]*1* \
                              (1+ TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] += 1
            self.cost+=self.state[index+1]*1* \
                              TRANSACTION_FEE_PERCENT
            self.trades+=1
        
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if not self.stop:
            if self.terminal:
                if self.highest_value < self.state[0]+self.state[1]*self.state[3] - self.min_delta:
                    self.last_improvement_index = 0
                    self.highest_value = self.state[0]+self.state[1]*self.state[3]
                else:
                    self.last_improvement_index += 1
                if self.epoch % 50 == 0:
                    logging.info("Epoch " + str(self.epoch) + ": " + str(self.state[0]+self.state[1]*self.state[3]))
                if self.last_improvement_index > self.require_improvement:
                    self.stop = True
                    logging.info("No improvement found during the last " + str(self.require_improvement) + " iterations, stopping optimization.")
                self.day += 1
                return self.state, self.reward, self.terminal,{}
            else:
                actions = actions * HMAX_NORMALIZE

                begin_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))

                argsort_actions = np.argsort(actions)

                sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
                buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

                for index in sell_index:
                    self._sell_stock(index, actions[index])

                for index in buy_index:
                    self._buy_stock(index, actions[index])
                self.day += 1
                self.data = self.df.loc[self.day,:]
                self.state = [self.state[0]] + \
                             self.data.adjcp.values.tolist() + \
                             list(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]) + \
                             self.data.gold.values.tolist() + \
                             self.data.interest.values.tolist() + \
                             self.data.index.values.tolist() + \
                             self.data.similar.values.tolist() + \
                             self.data.vix.values.tolist()
                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
                self.reward = end_total_asset - begin_total_asset
                self.asset_memory.append(end_total_asset)
                self.rewards_memory.append(self.reward)
                self.reward = self.reward*REWARD_SCALING
        else:
            self.day += 1

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        if not self.stop:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.data = self.df.loc[self.df.index[self.day],:]
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.day = 0
            self.rewards_memory = []
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                         self.data.adjcp.values.tolist() + \
                         [0] * STOCK_DIM + \
                         self.data.gold.values.tolist() + \
                         self.data.interest.values.tolist() + \
                         self.data.index.values.tolist() + \
                         self.data.similar.values.tolist() + \
                         self.data.vix.values.tolist()
            if self.epoch == 0 or self.epoch % 300*(len(self.df)//2) != 0:
                self.epoch += 1
            else:
                self.last_improvement_index = 0
                self.stop = False
                self.epoch = 0
                self.highest_value = 0
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]