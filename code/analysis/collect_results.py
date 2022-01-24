import os
import pandas as pd
import yfinance as yf
import numpy as np


file_path = "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/"

assets = [ 'AAPL','ORCL',
                        'ACN','MSFT','IBM','CSCO','NVDA','ADBE','HPQ','INTC','TESS','ASYS','CTG',
                       'BELFB','AVNW','LYTS','JPM','BAC','V','PFE','MRK','JNJ','CAJ','NICE','TSM','SNE','UMC','CHKP',
                       'SILC','GILT','TSEM','LFC','SMFG','SHG','NOK','ASML','ERIC','SAP','TEL','LOGI','HSBC','ING',

                       'BCS','0992.HK','3888.HK','0763.HK','0939.HK','2318.HK','0998.HK','GC=F','SI=F','PL=F','HG=F',
                       'CL=F','HO=F','NG=F','RB=F','HE=F','LE=F','GF=F','DC=F','ZC=F','ZS=F','KC=F','KE=F','DIA','SPY',
                       'QQQ','EWJ','EWT','EWY','EZU','CAC.PA','EXS1.DE','EXXY.MI','DBC']

def collect_results(agent):
    result = pd.DataFrame()
    for asset in assets:
        for run in ["run1","run2","run3"]:
            with os.scandir("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/" + agent + "/" + asset + "/" + run) as root_dir:
                for entry in root_dir:
                    if "final_results_1" in entry.name:
                        df = pd.read_csv(entry.path)
                        df["agent"] = agent
                        df["asset"] = asset
                        df["run"] = run
                        result = result.append(df)
    result.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent + "_final_results.csv")

def column_results(agent):
    df = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent +"_final_results.csv",
                     index_col=[0])
    try:
        result = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent +"_performance_analysis.csv",
                        index_col=[0])
    except FileNotFoundError:
        result = pd.DataFrame(columns=["value"])
    for column in df.columns.values:
        if "iteration" not in column and "agent" not in column and "run" not in column and "asset" not in column:
            result.loc["mean "+column] = df[column].mean()
            result.loc["std  deviation "+column] = df[column].std()

            for fold in ["fold1","fold2"]:
                result.loc["mean "+ column + " " + fold] = df.loc[df["iteration"].str.contains(fold),column].mean()
                result.loc["std deviation " +column +  " " + fold] = df.loc[df["iteration"].str.contains(fold), column].std()

    result.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent +"_performance_analysis.csv")

def split_in_sectors(agent):
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results_corona.csv",
        index_col=[0])
    for index, row in df.iterrows():
        asset_name = row["asset"]

        starting_date = '2010-01-02'
        ending_date = '2019-12-31'

        asset = yf.Ticker(asset_name)
        series = asset.history(start=starting_date, end=ending_date)["Close"]
        value_return =  series - series.shift(1)
        value_return = value_return.dropna()

        sp = yf.Ticker("^GSPC")
        series_sp = sp.history(start=starting_date, end=ending_date)["Close"]
        value_return_market = series_sp.shift(1) - series_sp
        value_return_market = value_return_market.dropna()

        if asset_name in ["ORCL", "AAPL", "ACN", "MSFT", "IBM", "CSCO", "NVDA", "ADBE", "HPQ", "INTC"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "usa"
            df.loc[df["asset"] == asset_name, "sector"] = "technology"
            df.loc[df["asset"] == asset_name, "company size"] = "market leader"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()

            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(value_return_market) / value_return_market.var()

        elif asset_name in ["TESS", "ASYS","CTG", "BELFB", "AVNW", "LYTS"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "usa"
            df.loc[df["asset"] == asset_name, "sector"] = "technology"
            df.loc[df["asset"] == asset_name, "company size"] = "small"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(value_return_market) / value_return_market.var()

        elif asset_name in ["JPM", "BAC", "V"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "usa"
            df.loc[df["asset"] == asset_name, "sector"] = "financials"
            df.loc[df["asset"] == asset_name, "company size"] = "market leader"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["PFE", "MRK", "JNJ"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "usa"
            df.loc[df["asset"] == asset_name, "sector"] = "healthcare"
            df.loc[df["asset"] == asset_name, "company size"] = "market leader"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["CAJ", "NICE", "TSM", "SNE", "UMC", "CHKP"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "asia"
            df.loc[df["asset"] == asset_name, "sector"] = "technology"
            df.loc[df["asset"] == asset_name, "company size"] = "market leader"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["SILC", "GILT", "TSEM"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "asia"
            df.loc[df["asset"] == asset_name, "sector"] = "technology"
            df.loc[df["asset"] == asset_name, "company size"] = "small"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["LFC", "SMFG", "SHG"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "asia"
            df.loc[df["asset"] == asset_name, "sector"] = "financials"
            df.loc[df["asset"] == asset_name, "company size"] = "market leader"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["NOK", "ASML", "ERIC", "SAP", "TEL", "LOGI"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "europe"
            df.loc[df["asset"] == asset_name, "sector"] = "technology"
            df.loc[df["asset"] == asset_name, "company size"] = "market leader"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["HSBC", "ING", "BCS"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "europe"
            df.loc[df["asset"] == asset_name, "sector"] = "financials"
            df.loc[df["asset"] == asset_name, "company size"] = "market leader"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["0992.HK", "3888.HK", "0763.HK"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "sehk"
            df.loc[df["asset"] == asset_name, "region"] = "asia"
            df.loc[df["asset"] == asset_name, "sector"] = "technology"
            df.loc[df["asset"] == asset_name, "company size"] = "market leader"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["0939.HK", "2318.HK", "0998.HK"]:
            df.loc[df["asset"] == asset_name, "class"] = "stock"
            df.loc[df["asset"] == asset_name, "exchange"] = "sehk"
            df.loc[df["asset"] == asset_name, "region"] = "asia"
            df.loc[df["asset"] == asset_name, "sector"] = "financials"
            df.loc[df["asset"] == asset_name, "company size"] = "market leader"
            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["GC=F", "SI=F", "PL=F", "HG=F"]:
            df.loc[df["asset"] == asset_name, "class"] = "commodity"
            df.loc[df["asset"] == asset_name, "exchange"] = "COMEX"

            df.loc[df["asset"] == asset_name, "sector"] = "metal"

            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["CL=F", "HO=F", "NG=F", "RB=F"]:
            df.loc[df["asset"] == asset_name, "class"] = "commodity"
            df.loc[df["asset"] == asset_name, "exchange"] = "COMEX"

            df.loc[df["asset"] == asset_name, "sector"] = "energy"

            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["HE=F", "LE=F", "GF=F","DC=F"]:
            df.loc[df["asset"] == asset_name, "class"] = "commodity"
            df.loc[df["asset"] == asset_name, "exchange"] = "COMEX"

            df.loc[df["asset"] == asset_name, "sector"] = "livestock and meat"

            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["ZC=F", "ZS=F", "KC=F", "KE=F"]:
            df.loc[df["asset"] == asset_name, "class"] = "commodity"
            df.loc[df["asset"] == asset_name, "exchange"] = "COMEX"

            df.loc[df["asset"] == asset_name, "sector"] = "agriculture"

            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["DIA", "SPY", "QQQ"]:
            df.loc[df["asset"] == asset_name, "class"] = "etf"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "usa"
            df.loc[df["asset"] == asset_name, "sector"] = "global indice"

            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["EWJ", "EWT", "EWY"]:
            df.loc[df["asset"] == asset_name, "class"] = "etf"
            df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"
            df.loc[df["asset"] == asset_name, "region"] = "asia"
            df.loc[df["asset"] == asset_name, "sector"] = "global indice"

            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["EZU", "CAC.PA", "EXS1.DE"]:
            df.loc[df["asset"] == asset_name, "class"] = "etf"
            if asset_name in "EXS1.DE":
                df.loc[df["asset"] == asset_name, "exchange"] = "XETRA"
            elif asset_name in "EZU":
                df.loc[df["asset"] == asset_name, "exchange"] = "BATS"
            else:
                df.loc[df["asset"] == asset_name, "exchange"] = "PSE"

            df.loc[df["asset"] == asset_name, "region"] = "europe"
            df.loc[df["asset"] == asset_name, "sector"] = "global indice"

            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        elif asset_name in ["EXXY.MI", "DBC"]:
            df.loc[df["asset"] == asset_name, "class"] = "etf"
            if asset_name in ["EXXY.MI"]:
                df.loc[df["asset"] == asset_name, "exchange"] = "MSE"
            else:
                df.loc[df["asset"] == asset_name, "exchange"] = "nyse_nasdaq"

            df.loc[df["asset"] == asset_name, "sector"] = "commodity"

            df.loc[df["asset"] == asset_name, "volatility asset"] = value_return.std()
            df.loc[df["asset"] == asset_name, "mean difference asset"] = value_return.mean()
            df.loc[df["asset"] == asset_name, "beta"] = value_return.cov(
                value_return_market) / value_return_market.var()

        if asset_name in ["AAPL","GOOG","V","NICE","ASML"]:
            df.loc[df["asset"] == asset_name, "course development"] = "increasing rapidly"
        elif asset_name in ["HSBC","0998.HK","TESS","AVNW","EXXY.MI"]:
            df.loc[df["asset"] == asset_name, "course development"] = "decreasing rapidly"

    df.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results_corona.csv")

def hyperparameter_results():
    result = pd.DataFrame()
    for agent in ["q_learning_agent","actor_critic_agent","duel_recurrent_q_learning_agent", "DDPG"]:
        for asset in assets:
            for run in ["run1","run2","run3"]:
                if agent == "DDPG":
                    with os.scandir("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository2/" + agent + "/" + asset + "/" + run) as root_dir:
                        for entry in root_dir:
                            if "hyperparameter" in entry.name:
                                df = pd.read_csv(entry.path, index_col=[0])
                                df["agent"] = agent
                                df["asset"] = asset
                                result = result.append(df)
                else:
                    with os.scandir("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/" + agent + "/" + asset + "/" + run) as root_dir:
                        for entry in root_dir:
                            if "hyperparameter" in entry.name:
                                df = pd.read_csv(entry.path,index_col=[0])
                                df["agent"] = agent
                                df["asset"] = asset
                                result = result.append(df)
    result.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_hyperparameters.csv")

def portfolio_movement_results():
    beta_df = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv",index_col=[0])
    result = pd.DataFrame()
    for agent in ["q_learning_agent", "actor_critic_agent", "duel_recurrent_q_learning_agent", "DDPG"]:
        for asset in assets:
            beta_factor = beta_df[beta_df["asset"] == asset]["beta"].mean()
            #grouping of beta_factor in 33th and 66th percentils
            if beta_factor<0.007478297588004964:
                beta_factor=1
            elif beta_factor<0.018493537526106555:
                beta_factor=2
            else:
                beta_factor=3
            for run in ["run1", "run2", "run3"]:
                with os.scandir(
                        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/" + agent + "/" + asset + "/" + run) as root_dir:
                    for entry in root_dir:
                        if "portfolio_movement" in entry.name:
                            df = pd.read_csv(entry.path, index_col=[0])
                            string = agent+"_"+asset+"_"+run
                            if len(df.columns) > 1:
                                result[string+"_fold1_"+str(beta_factor)] = df.iloc[:,0]
                                result[string+"_fold2_"+str(beta_factor)] = df.iloc[:,1]
    result.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_portfolio_movements.csv")

def quantile_of_clustering(cluster):
    df = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv",index_col=[0])
    print(np.quantile(df[cluster].unique(),2/3))

def merge_final_results():
    result = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv",index_col=[0])
    for agent in ["DDPG"]:
        if agent == "DDPG":
            df = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent + "_performance_analysis.csv",index_col=[0])
        else:
            df = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent + "_performance_analysis.csv",index_col=[0])
        df = df.reset_index(drop=True)
        result = result.append(df)
        print(result[result["agent"] == "DDPG"])
    result.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv")

def corona_results():
    assets = ["CL=F", "EWY", "DIA", "JPM", "DC=F", "V", "INTC", "SMFG", "EZU", "NICE"]
    result = pd.DataFrame()
    for agent in ["q_learning_agent",
        "actor_critic_agent",
        "turtle_agent",
        "moving_average_agent",
        "duel_recurrent_q_learning_agent",
        "DDPG"]:
        for asset in assets:
            if "moving" in agent or "turtle" in agent:
                runs = ["run4", "run5", "run6"]
            else:
                runs = ["run1", "run2", "run3"]
            for run in runs:
                if agent == "DDPG":
                    string = "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository2/" + agent + "/" + asset + "/" + run
                else:
                    string = "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/" + agent + "/" + asset + "/" + run
                with os.scandir(string) as root_dir:
                    for entry in root_dir:
                        if "final_results" in entry.name:
                            df = pd.read_csv(entry.path)
                            if len(df)<2:
                                df["agent"] = agent
                                df["asset"] = asset
                                df["run"] = run
                                result = result.append(df)
    result.to_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results_corona.csv")

if __name__ == "__main__":
    pass