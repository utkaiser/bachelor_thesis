import numpy as np
import pandas as pd
import asset_trading_agents.repository_1.agent as agent
import logging

class Agent(agent.Abstract_Agent):
    def __init__(self, trading_env,decay_rate,layer_size,gamma, memory_size):
        self.name = "moving_average_agent"
        super().__init__(trading_env)
        self.max_buy = 1
        self.max_sell = 1

    def buy(self, df_test):

        """
        real_movement = actual movement in the real world
        delay = how much interval you want to delay to change our decision from buy to sell, vice versa
        initial_state = 1 is buy, 0 is sell
        initial_money = 1000, ignore what kind of currency
        max_buy = max quantity for share to buy
        max_sell = max quantity for share to sell
        """

        short_window = int(0.025 * len(df_test))
        long_window = int(0.05 * len(df_test))

        signals = pd.DataFrame(index=df_test.index)
        signals['signal'] = 0.0

        signals['short_ma'] = df_test.rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_ma'] = df_test.rolling(window=long_window, min_periods=1, center=False).mean()

        signals['signal'][short_window:] = np.where(signals['short_ma'][short_window:]
                                                    > signals['long_ma'][short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()

        current_cash = self.initial_money
        states_sell = []
        states_buy = []
        current_inventory = 0
        portfolio_movement = pd.Series()
        transaction_sum = 0
        signal = signals['positions']

        def buy_stock(i, current_cash, current_inventory, transaction_sum):
            shares = current_cash // df_test[i]
            if shares < 1:
                logging.info(
                    'day %d: total cash %f, not enough money to buy a unit price %f'
                    % (i, current_cash, df_test[i])
                )

            else:
                if shares > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = shares
                current_cash -= buy_units * df_test[i] * (1+self.transaction_cost)
                transaction_sum += buy_units * df_test[i] * self.transaction_cost
                current_inventory += buy_units

                logging.info(
                    'day %d: buy %d units at price %f, current portfolio value %f'
                    % (i, buy_units, buy_units * df_test[i], current_cash + current_inventory * df_test[i])
                )
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
                    portfolio_movement.at[df_test.index[i]] = current_cash + current_inventory * df_test[i]
                else:
                    if current_inventory > self.max_sell:
                        sell_units = self.max_sell
                    else:
                        sell_units = current_inventory
                    current_inventory -= sell_units
                    current_cash += sell_units * df_test[i] * (1 - self.transaction_cost)
                    transaction_sum += sell_units * df_test[i] * self.transaction_cost
                    try:
                        invest = (
                            (df_test[i] - df_test[states_buy[-1]])
                            / df_test[states_buy[-1]]
                        ) * 100
                    except:
                        invest = 0
                    logging.info(
                        'day %d, sell %d units at price %f, investment %f %%, current portfolio value %f,'
                        % (i, sell_units, sell_units * df_test[i], invest, current_cash + current_inventory * df_test[i])
                    )
                states_sell.append(i)
            portfolio_movement.at[df_test.index[i]] = current_cash + current_inventory * df_test[i]
        invest = ((current_cash - self.initial_money) / self.initial_money) * 100
        total_gains = current_cash - self.initial_money + current_inventory * df_test[i]
        logging.info("Total transaction cost: %f" % (transaction_sum))
        return states_buy, states_sell, total_gains, invest, portfolio_movement



