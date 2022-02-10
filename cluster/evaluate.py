import scripts.help_functions_agents as prep_agents
import scripts.visualization as visualize
import scripts.trading_environment as trading_environment
import scripts.trading_performance as trading_performance
from matplotlib.backends.backend_pdf import PdfPages
import scripts.split_data as split
import logging
import pathlib
import sys
import pandas as pd

def evaluate(asset, agent_to_evaluate,run,repository="repository1"):

    '''
        Goal: run evaluation of selected agent on stock
        Author: Luis Kaiser, University of Wuerzburg, Bachelor Thesis

        Input:
        - asset: relevant asset to trade with
        - agent_to_evaluate: agent to test in this evaluation with specific asset
        - repository: repository to test

        Output:
        - plots of evaluation /results/repository1/<<evaluation_name>>/plots/<<asset>>.pdf with results
            (each asset has its own file, one file contains results of all relevant agents trading on the same asset)
        - log-file with documentation of process /results/repository1/<<evaluation_name>>/logfile.log
        - csv-file with brief results in /results/repository1/<<evaluation_name>>/results.csv
          performance measures:  - profit
                                 - volatility (standard deviation of portfolio_movement)
                                 - greatest_loss (most negative portfolio_movement)
                                 - sharpe_ratio (profit/volatility)
        - csv-file with portfolio_movement

        Description:
        create trading environment,
        download and preprocess data from yfinance,
        iterating over columns of prepared dataframe, iterating over number of runs,
        run 3 fold cross validation with 251 trading days (approximation of trading days per year) to test the agent,
        saving and visualizing the results per column
    '''

    # create trading environment
    Trading_env = trading_environment.Trading_env(
        evaluation_name=repository + "/" + agent_to_evaluate + "/" + asset + "/run"+str(run),
        relevant_assets=asset,
        file=__file__,
        ending_date = '2020-12-31'
    )

    hyperparameter_results = []
    final_results = []
    portfolio_movement_dict = {} #dict to store portfolio movements from test

    #get dataframe with asset
    df = Trading_env.relevant_data[asset].dropna()
    # fold_indices = [pd.Timestamp('2017-01-03 00:00:00'), pd.Timestamp('2018-01-02 00:00:00'), pd.Timestamp('2019-01-02 00:00:00')]
    # df_list = split.manual_split(df, [1762, 2013, 2264])

    fold_indices = [pd.Timestamp('2019-01-02 00:00:00'), pd.Timestamp('2020-01-02 00:00:00')]
    df_list = split.manual_split(df, [2264,2516])

    #create pdf
    i = 1
    while i > 0:
        p = pathlib.Path(Trading_env.result_dir + "/plot" + str(i) + ".pdf")
        if not p.is_file():
            break
        i += 1
    pdf = PdfPages(Trading_env.result_dir + "/plot" + str(i) + ".pdf")
    pdf.savefig(visualize.create_plot_after_preprocessing(df,
                                                          asset,
                                                          fold_indices))


    if agent_to_evaluate in ["moving_average_agent", "turtle_trading_agent"]:
        Agent = prep_agents.get_agent_by_name(agent_to_evaluate,
                                              trading_env=Trading_env)
        fold_number = 1
        for df_training, df_validation, df_test in df_list:
            iteration_name = "fold" + str(fold_number)
            column_results = [iteration_name]
            states_buy, states_sell, total_gains, gains_percentage, portfolio_movement = Agent.buy(df_test=df_test)
            volatility, greatest_loss, sharpe_ratio = trading_performance.get_performance(portfolio_movement)
            column_results += [total_gains, volatility, greatest_loss, sharpe_ratio]
            pdf.savefig(visualize.create_plot_result(df_test,
                                                     iteration_name,
                                                     states_buy,
                                                     states_sell,
                                                     total_gains,
                                                     gains_percentage))
            column_results += [total_gains, volatility, greatest_loss, sharpe_ratio]
            final_results.append(column_results)
            fold_number += 1
            portfolio_movement_dict[iteration_name] = portfolio_movement

    else:
        param_grid = {
            'decay_rate': [0.01],#,0.001
            'layer_size': [256],#,512
            'memory_size': [251],#one year or three years,753
            'gamma': [0.9]#,0.999
        }
        fold_number = 0
        for df_training, df_validation, df_test in df_list:
            fold_number += 1
            highest_profit = -10000
            best_decay_rate = 0
            best_memory_size = 0
            best_layer_size = 0
            best_gamma = 0
            best_iteration = ""
            for decay_rate in param_grid["decay_rate"]:
                for layer_size in param_grid["layer_size"]:
                    for memory_size in param_grid["memory_size"]:
                        for gamma in param_grid["gamma"]:
                            Trading_env.memory_size = memory_size
                            Trading_env.layer_size = layer_size
                            Trading_env.gamma = gamma
                            Trading_env.decay_rate = decay_rate
                            iteration_name = "fold" + str(fold_number) + "___" + str(decay_rate) + "_" +str(layer_size) + "_" + str(memory_size) + "_" + str(gamma)
                            Agent = prep_agents.get_agent_by_name(agent_to_evaluate,
                                                                  trading_env=Trading_env,
                                                                  decay_rate=decay_rate,
                                                                  layer_size=layer_size,
                                                                  gamma=gamma,
                                                                  memory_size=memory_size)

                            column_results = [iteration_name]
                            logging.info("Start Training of " + iteration_name + ".")
                            Agent.train(df_training=df_training.values.tolist())
                            logging.info(("Start evaluating hyperparameters of " + iteration_name + "."))
                            states_buy, states_sell, total_gains, gains_percentage, portfolio_movement = Agent.buy(
                                df_test=df_validation.values.tolist())
                            Agent.sess.close()
                            volatility, greatest_loss, sharpe_ratio = trading_performance.get_performance(
                                portfolio_movement)
                            column_results += [total_gains, volatility, greatest_loss, sharpe_ratio]
                            hyperparameter_results.append(column_results)

                            if total_gains > highest_profit:
                                highest_profit = total_gains
                                best_gamma = gamma
                                best_layer_size = layer_size
                                best_decay_rate = decay_rate
                                best_memory_size = memory_size
                                best_iteration = iteration_name

            logging.info("Start Testing of " + best_iteration + ".")
            column_results = [best_iteration]
            Agent = prep_agents.get_agent_by_name(agent_to_evaluate,
                                                  trading_env=Trading_env,
                                                  decay_rate=best_decay_rate,
                                                  layer_size=best_layer_size,
                                                  gamma=best_gamma,
                                                  memory_size=best_memory_size)
            Agent.train(df_training=df_training.values.tolist()+df_validation.values.tolist())
            states_buy, states_sell, total_gains, gains_percentage, portfolio_movement = Agent.buy(
                df_test=df_test.values.tolist())
            Agent.sess.close()
            volatility, greatest_loss, sharpe_ratio = trading_performance.get_performance(
                portfolio_movement)
            column_results += [total_gains, volatility, greatest_loss, sharpe_ratio]
            portfolio_movement_dict[best_iteration] = portfolio_movement
            final_results.append(column_results)
            pdf.savefig(visualize.create_plot_result(df_test,
                                                     best_iteration,
                                                     states_buy,
                                                     states_sell,
                                                     total_gains,
                                                     gains_percentage))
    pdf.close()

    trading_performance.save_performance(final_results, Trading_env.result_dir + "/final_results_")
    trading_performance.save_performance(hyperparameter_results, Trading_env.result_dir + "/hyperparameter_results_")
    trading_performance.save_portfolio_movement(portfolio_movement_dict, Trading_env.result_dir + "/portfolio_movements_")

if __name__ == "__main__":
    asset = sys.argv[2]
    run = sys.argv[4]
    agent = sys.argv[6]
    # agent = "q_learning_agent"
    # asset = "AAPL"
    # run = 7
    evaluate(
        agent_to_evaluate=agent,
        asset=asset,
        run=run
    )




