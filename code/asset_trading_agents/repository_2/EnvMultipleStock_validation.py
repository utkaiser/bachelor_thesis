import numpy as np
from gym.utils import seeding
import gym
from gym import spaces

HMAX_NORMALIZE = 1
INITIAL_ACCOUNT_BALANCE=100000
STOCK_DIM = 2
TRANSACTION_FEE_PERCENT = 0.01
REWARD_SCALING = 1e-4

class StockEnvValidation(gym.Env):

    def __init__(self, df,asset_name):
        self.asset_name = asset_name
        self.day = 0
        self.df = df
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (15,))
        self.data = self.df.loc[df.index[self.day],:]
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
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self._seed()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal:
            #last day
            return self.state, self.reward, self.terminal,{}
        else:
            actions = actions * HMAX_NORMALIZE
            begin_total_asset = self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index)

            for index in buy_index:
                self._buy_stock(index)

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
            
            end_total_asset = self.state[0]+ sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory.append(end_total_asset)
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*REWARD_SCALING
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.df.index[self.day],:]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.gold.values.tolist() + \
                     self.data.interest.values.tolist() + \
                     self.data.index.values.tolist() + \
                     self.data.similar.values.tolist() + \
                     self.data.vix.values.tolist()
        return self.state
    
    def render(self, mode='human',close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _sell_stock(self, index):
        # perform sell action based on the sign of the action
        if self.state[index + STOCK_DIM + 1] > 0 and index == 0:
            # update balance
            self.state[0] += \
                self.state[index + 1] * 1 * \
                (1 - TRANSACTION_FEE_PERCENT)

            self.state[index + STOCK_DIM + 1] -= 1
            self.cost += self.state[index + 1] * 1 * \
                         TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass

    def _buy_stock(self, index):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index + 1]
        if available_amount>1 and index == 0:
            # update balance
            self.state[0] -= self.state[index + 1] * 1 * \
                             (1 + TRANSACTION_FEE_PERCENT)

            self.state[index + STOCK_DIM + 1] += 1

            self.cost += self.state[index + 1] * 1 * \
                         TRANSACTION_FEE_PERCENT
            self.trades += 1