from abc import ABC, abstractmethod
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from collections import deque
import logging

class Abstract_Agent(ABC):

    @abstractmethod
    def __init__(self, trading_env):
        # simple variabel definitions
        tf.compat.v1.disable_eager_execution()
        self.state_size = trading_env.state_size
        self.transaction_cost = trading_env.transaction_cost
        self.initial_money = trading_env.initial_money
        self.iterations = trading_env.iterations
        self.checkpoint = trading_env.checkpoint
        self.memory = deque()
        self.output_size = trading_env.output_size
        self.min_epsilon = trading_env.min_epsilon
        self.action_size = trading_env.action_size
        self.action = 0
        self.starting_money = self.initial_money
        self.current_cash = self.initial_money
        self.total_profit = 0
        self.transaction_sum = 0
        self.states_sell = []
        self.states_buy = []
        self.inventory = []
        self.portfolio_movement = []
        self.next_state = None
        self.starting_date = trading_env.starting_date

        #early stopping
        self.require_improvement = trading_env.require_improvement
        self.last_improvement_index = 0 #if index>15, then break
        self.highest_value = 0
        self.best_session = None
        self.min_delta = 0 #minimum change in the monitored quantity to qualify as an improvement
        self.continue_loop = True



#Training
    def log_current_epoch(self,epoch,cost, portfolio_value):
        logging.info('epoch: %d, cost: %f, total value portfolio: %f' % (epoch,
                                                                         cost,
                                                                         portfolio_value))

    def reset_training_variables(self):
        self.total_profit = 0
        self.inventory = []
        self.current_cash = self.initial_money
        self.starting_money = self.current_cash

    def take_action_training(self, t, df_training):
        # buy
        if self.action == 1 and self.starting_money >= df_training[t] and t < (len(df_training) - 15):
            self.inventory.append(df_training[t])
            self.current_cash -= df_training[t] * (1+self.transaction_cost)

        # sell
        elif self.action == 2 and len(self.inventory):
            del self.inventory[0]
            self.current_cash += df_training[t] * (1-self.transaction_cost)
        self.starting_money = len(self.inventory) * df_training[t] + self.current_cash
        logging.info("inventory" + str(len(self.inventory)))


    def early_stopping(self, value):
        if self.highest_value < value - self.min_delta:
            self.last_improvement_index = 0
            self.highest_value = value
            self.best_session = self.sess
        else:
            self.last_improvement_index += 1

        if self.last_improvement_index > self.require_improvement:
            # break loop
            self.sess = self.best_session
            logging.info("No improvement found during the last " + str(
                self.require_improvement) + " iterations, stopping optimization.")
            return True
        else:
            return False


#testing
    def reset_test_variables(self):
        self.transaction_sum = 0
        self.portfolio_movement = []
        self.current_cash = self.initial_money
        self.states_sell = []
        self.states_buy = []
        self.inventory = []


    def take_action_testing(self,t,df_test):
        # buy
        if self.action == 1 and self.current_cash >= df_test[t] and t < (len(df_test) - 15):
            self.inventory.append(df_test[t])
            self.current_cash -= df_test[t] * (1+self.transaction_cost)
            self.transaction_sum += df_test[t] * self.transaction_cost
            self.states_buy.append(t)

        # sell
        elif self.action == 2 and len(self.inventory):
            bought_price = self.inventory.pop(0)
            self.current_cash += df_test[t] * (1-self.transaction_cost)
            self.transaction_sum += df_test[t] * self.transaction_cost
            self.states_sell.append(t)

        self.portfolio_movement.append(self.current_cash + len(self.inventory) * df_test[t])
        self.state = self.next_state
        logging.info("inventory" + str(len(self.inventory)))

    def determine_results(self, last_price):
        self.invest = ((self.current_cash - self.initial_money) / self.initial_money) * 100
        self.total_gains = self.current_cash - self.initial_money + len(self.inventory) * last_price
        logging.info("Total transaction cost: %f" % (self.transaction_sum))


