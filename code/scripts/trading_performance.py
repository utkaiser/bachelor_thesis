import numpy as np
import pandas as pd
import logging
import pathlib
import os
import scripts.visualization as visualization
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf

def get_performance(portfolio_movement):
    '''
        Goal: display diverse information about performance of agent

        Input:
        - account_movement: account balance movement over trading time

        Output:
        - 1: Standard derivation of portfolio value movements, displays volatility
        - 2: greatest loss, descrease of portfolio value
        - 3: Sharpe ratio, combines volatility and monetary result
    '''
    return np.std(portfolio_movement), determine_greatest_loss(portfolio_movement), determine_sharpe_ratio(portfolio_movement)


def determine_greatest_loss(portfolio_movement):
    movement = [portfolio_movement[i+1] - portfolio_movement[i] for i in range(0,len(portfolio_movement)-2)]
    return min(movement)


def determine_sharpe_ratio(portfolio_movement):
    movement = [portfolio_movement[i+1] - portfolio_movement[i] for i in range(0,len(portfolio_movement)-2)]
    mean_return = sum(movement) / len(movement)
    volatility = np.std(portfolio_movement)
    if mean_return != 0 and volatility != 0:
        #assumptions: 252 trading days in 1 year
        sharpe_ratio = np.sqrt(252) * mean_return / volatility
    else:
        sharpe_ratio = 0
    return sharpe_ratio

#anderer speicherort noch??????
def save_performance(results, result_dir):
    results = pd.DataFrame(results)
    results.columns = ["iteration", "total gains", "volatility", "greatest loss", "sharpe ratio"]
    results.set_index("iteration", inplace=True)

    i = 1
    while i > 0:
        p = pathlib.Path(result_dir + str(i) + ".csv")
        if not p.is_file():
            break
        i += 1

    results.to_csv(result_dir + str(i) + ".csv")


def save_portfolio_movement(movements, result_dir):
    logging.info("Save portfolio movements in a csv-file.")
    movements = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in movements.items()]))

    i = 1
    while i > 0:
        p = pathlib.Path(result_dir + str(i) + ".csv")
        if not p.is_file():
            break
        i += 1

    movements.to_csv(result_dir + str(i) + ".csv")

def merge_results():
    agents = [
        # "moving_average_agent",
        # "turtle_agent",
        "actor_critic_agent",
        "actor_critic_duel_agent",
        "actor_critic_duel_recurrent_agent",
        "actor_critic_recurrent_agent",
        "curiosity_q_learning_agent",
        "double_duel_q_learning_agent",
        "double_duel_recurrent_q_learning_agent",
        "double_q_learning_agent",
        "double_recurrent_q_learning_agent",
        "duel_curiosity_q_learning_agent",
        "duel_q_learning_agent",
        "duel_recurrent_q_learning_agent",
        "policy_gradient_agent",
        "q_learning_agent",
        "recurrent_curiosity_q_learning_agent",
        "recurrent_q_learning_agent",
    ]
    result = pd.read_csv("/results/repository1/preselection_and_hyperparameter/all_results.csv", index_col=0)
    for agent in agents:
        with os.scandir("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/"+agent) as root_dir:
            for entry in root_dir:
                if "results" in entry.name:
                    df = pd.read_csv(entry.path)
                    print(df)
                    result = result.append(df, ignore_index=True)
    print(result)
    result.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/all_results.csv")

def calculate_average_performance():
    agents = [
        "moving_average_agent",
        "turtle_agent",
        "actor_critic_agent",
        "actor_critic_duel_agent",
        "actor_critic_duel_recurrent_agent",
        "actor_critic_recurrent_agent",
        "curiosity_q_learning_agent",
        "double_duel_q_learning_agent",
        "double_duel_recurrent_q_learning_agent",
        "double_q_learning_agent",
        "double_recurrent_q_learning_agent",
        "duel_curiosity_q_learning_agent",
        "duel_q_learning_agent",
        "duel_recurrent_q_learning_agent",
        "policy_gradient_agent",
        "q_learning_agent",
        "recurrent_curiosity_q_learning_agent",
        "recurrent_q_learning_agent",
    ]
    results = []
    for agent in agents:
        with os.scandir("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/"+agent) as root_dir:
            for entry in root_dir:
                if "results" in entry.name:
                    df = pd.read_csv(entry.path)
                    results.append([agent,df["total gains"].mean(), df["volatility"].mean(),df["greatest loss"].mean(),df["sharpe ratio"].mean()])
    result_df = pd.DataFrame(results)
    result_df.columns = ["agent", "total gains average", "volatility average", "greatest loss average", "sharpe ratio average"]
    result_df.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/all_results_mean.csv")

def get_best_agent():
    df = pd.read_csv("/results/repository1/preselection_and_hyperparameter/all_results_mean.csv", index_col=0)
    ranking = pd.DataFrame()
    for attribute in ["total gains average", "volatility average", "greatest loss average", "sharpe ratio average"]:
        df.sort_values(attribute, inplace=True)
        if attribute != "volatility average":
            df = df.iloc[::-1]
        df.index = np.arange(1, len(df) + 1)
        ranking[attribute] = df["agent"]
    ranking.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/all_results_ranking.csv")

def sum_hyperparameters():
    agents = [
        "actor_critic_agent",
        "double_duel_recurrent_q_learning_agent",
        "duel_recurrent_q_learning_agent",
        "q_learning_agent",
        "recurrent_q_learning_agent",
    ]
    results = pd.DataFrame()
    for agent in agents:
        with os.scandir(
                "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/" + agent + "/hyperparameter") as root_dir:
            for entry in root_dir:
                if "best_parameters" in entry.name:
                    if len(results) == 0:
                        results = pd.read_csv(entry.path)
                    else:
                        df = pd.read_csv(entry.path)
                        results = results.append(df)
    results.to_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/all_hyperparameters.csv")

def unique_function(by=["batch_size"]):
    df = pd.read_csv("/results/repository1/preselection_and_hyperparameter/all_hyperparameters.csv")
    df = df.groupby(by).size().reset_index().rename(columns={0: 'count'})
    df.to_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/all_hyperparameters_count_.csv")

def group_portfolio_movements():
    agents = [
        "actor_critic_agent",
    #     "double_duel_recurrent_q_learning_agent",
    #     "duel_recurrent_q_learning_agent",
        "q_learning_agent",
    #     "recurrent_q_learning_agent",
    ]
    results = pd.DataFrame()
    for agent in agents:
        with os.scandir(
                "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/" + agent) as root_dir:
            for entry in root_dir:
                if "portfolio_movements" in entry.name:
                    if len(results) == 0:
                        results = pd.read_csv(entry.path,index_col=0)
                    else:
                        df = pd.read_csv(entry.path,index_col=0)
                        results = pd.concat([df,results],axis=1)

    results.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/all_portfolio_movements.csv")

def display_portfolio_movements(agents,asset_ticker, fold=3, iteration=1):

    if fold == 1:
        starting_date = '2016-12-28'
        ending_date = '2017-12-27'
    elif fold == 2:
        starting_date = '2017-12-28'
        ending_date = '2018-12-28'
    else:
        starting_date = '2018-12-31'
        ending_date = '2019-12-30'


    asset = yf.Ticker(asset_ticker)
    series = asset.history(start=starting_date, end=ending_date)["Close"]

    df = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/all_portfolio_movements.csv",index_col=0)
    pdf = PdfPages("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/portfolio_movement_" + asset_ticker + ".pdf")
    selected_movements = pd.DataFrame()
    for column_name in df.columns:
        for agent_name in agents:
            if agent_name+"_"+asset_ticker+"_iteration"+str(fold)+"_fold"+str(iteration) in column_name:
                selected_movements[agent_name] = df[column_name]
    selected_movements.index = series.index
    pdf.savefig(visualization.create_plot_portfolio_movement(selected_movements, asset_ticker))
    pdf.close()

if __name__ == "__main__":
    #group_portfolio_movements()
    display_portfolio_movements(["q_learning_agent", "actor_critic_agent"], "AAPL")



