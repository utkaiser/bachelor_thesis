import numpy as np
import pandas as pd
import logging
import asset_trading_agents.repository_1.agent as agent

class Agent(agent.Abstract_Agent):
    def __init__(self, trading_env,decay_rate,layer_size,gamma, memory_size):
        self.name = "turtle_trading_agent"
        super().__init__(trading_env)
        self.max_buy = 1
        self.max_sell = 1

    def buy(self, df_test):

        count = int(np.ceil(len(df_test) * 0.1))
        signals = pd.DataFrame(index=df_test.index)
        signals['signal'] = 0.0
        signals['trend'] = df_test
        signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
        signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
        signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
        signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1
        pd.set_option('display.max_rows', 252)
        signal = signals['signal']

        current_cash = self.initial_money
        states_sell = []
        states_buy = []
        current_inventory = 0
        portfolio_movement = []
        transaction_sum = 0

        def buy_stock(i, current_cash, current_inventory, transaction_sum):
            shares = current_cash // df_test[i]
            if shares < 1:
                logging.info(
                    'day %d: total cash %f, not enough money to buy a unit price %f'
                    % (i, current_cash, df_test[i])
                )
                portfolio_movement.append(current_cash + current_inventory * df_test[i])
            else:
                if shares > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = shares
                current_cash -= buy_units * df_test[i] * (1+self.transaction_cost)
                transaction_sum += buy_units * df_test[i] * self.transaction_cost
                portfolio_movement.append(current_cash + current_inventory * df_test[i])
                current_inventory += buy_units
                logging.info('day %d: buy %d units at price %f, total balance %f'% (i, buy_units, buy_units * df_test[i], current_cash + current_inventory * df_test[i]))
                states_buy.append(0)
            return current_cash, current_inventory, transaction_sum

        i=0
        for i in range(df_test.shape[0] - int(0.025 * len(df_test))):
            state = signal[i]
            if state == 1:
                current_cash, current_inventory, transaction_sum = buy_stock(i, current_cash, current_inventory, transaction_sum)
                states_buy.append(i)
            elif state == -1:
                if current_inventory == 0:
                        logging.info('day %d: cannot sell anything, inventory 0' % (i))
                else:
                    if current_inventory > self.max_sell:
                        sell_units = self.max_sell
                    else:
                        sell_units = current_inventory
                    current_inventory -= sell_units
                    current_cash += sell_units * df_test[i] * (1 - self.transaction_cost)
                    transaction_sum += sell_units * df_test[i] * self.transaction_cost
                    portfolio_movement.append(current_cash + current_inventory * df_test[i])
                    try:
                        invest = (
                            (df_test[i] - df_test[states_buy[-1]])
                            / df_test[states_buy[-1]]
                        ) * 100
                    except:
                        invest = 0
                    logging.info(
                        'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                        % (i, sell_units, sell_units * df_test[i], invest, current_cash + current_inventory * df_test[i])
                    )
                states_sell.append(i)

        invest = ((current_cash - self.initial_money) / self.initial_money) * 100
        total_gains = current_cash - self.initial_money + current_inventory * df_test[i]
        logging.info("Total transaction cost: %f" % (transaction_sum))
        return states_buy, states_sell, total_gains, invest, portfolio_movement
