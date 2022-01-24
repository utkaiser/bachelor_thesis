import numpy as np
from gym.utils import seeding
import gym
from gym import spaces

max_shares_buy_sell = 1
INITIAL_ACCOUNT_BALANCE=100000
STOCK_DIM = 2
TRANSACTION_FEE_PERCENT = 0.01
REWARD_SCALING = 1e-4

class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0,turbulence_threshold=140
                 ,initial=True, previous_state=[], model_name='', iteration=''):
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
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
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self._seed()
        self.model_name=model_name        
        self.iteration=iteration


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action

        if self.state[index+STOCK_DIM+1] > 0 and index+0 == 0:
            #update balance
            self.state[0] += \
            self.state[index+1]*1 * \
             (1- TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] -= 1
            self.cost +=self.state[index+1]*1 * \
             TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:
            pass

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]

        if index+1 == 1 and available_amount>1:
            #update balance
            self.state[0] -= self.state[index+1]*1* \
                              (1+ TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] += 1

            self.cost+=self.state[index+1]*1* \
                              TRANSACTION_FEE_PERCENT
            self.trades+=1

        
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal:
            #last day
            return self.state, self.reward, self.terminal,{}

        else:
            actions = actions * max_shares_buy_sell
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
            self.asset_memory.append(end_total_asset)
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*REWARD_SCALING
        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
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
        else:
            previous_total_asset = self.previous_state[0]+ \
            sum(np.array(self.previous_state[1:(STOCK_DIM+1)])*np.array(self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory = [previous_total_asset]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []
            self.state = [self.previous_state[0]] + \
                         self.data.adjcp.values.tolist() + \
                         self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)] + \
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