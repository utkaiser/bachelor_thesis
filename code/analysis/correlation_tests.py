import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

agent_list = ["q_learning_agent",
                      "actor_critic_agent",
                      "turtle_agent",
                      "moving_average_agent",
                      "duel_recurrent_q_learning_agent",
                      "ddpg_agent",
                      "improved_q_learning_agent"
                      ]

asset_list = ['AAPL',
                       'ORCL','ACN','MSFT','IBM','CSCO','NVDA','ADBE','HPQ','INTC','TESS','ASYS','CTG',
                       'BELFB','AVNW','LYTS','JPM','BAC','V','PFE','MRK','JNJ','CAJ','NICE','TSM','SNE','UMC','CHKP',
                       'SILC','GILT','TSEM','LFC','SMFG','SHG','NOK','ASML','ERIC','SAP','TEL','LOGI','HSBC','ING',

                       'BCS','0992.HK','3888.HK','0763.HK','0939.HK','2318.HK','0998.HK','GC=F','SI=F','PL=F','HG=F',
                       'CL=F','HO=F','NG=F','RB=F','HE=F','LE=F','GF=F',
        'DC=F',
        'ZC=F','ZS=F','KC=F','KE=F','DIA','SPY',
                       'QQQ','EWJ','EWT','EWY','EZU','CAC.PA','EXS1.DE','EXXY.MI','DBC'
               ]

def correlation_matrix():
    mean_df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv",
        index_col=[0])
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/improved_q_learning_agent_final_results.csv",
        index_col=[0])

    means = {}
    for index, row in mean_df.iterrows():
        if not pd.isnull(row["volatility asset"]):
            means[row["asset"]] = row["mean difference asset"]

    average_result = []
    volatility = []
    for agent in ["improved_q_learning_agent"]:
        df_agent = df[df["agent"] == agent]
        #quantils = pd.Series(df_agent["beta"].unique()).quantile([.25, .5, .75]).values.tolist()
        #df_agent = df_agent[df_agent["beta"] <= quantils[0]]
        for asset in df["asset"].unique():
            if not pd.isnull(df_agent[df_agent["asset"] == asset]["total gains"].dropna().mean()):
                average_result.append(df_agent[df_agent["asset"] == asset]["total gains"].dropna().mean())
                #volatility.append(df_agent[df_agent["asset"] == asset]["beta"].dropna().mean())
                volatility.append(means[asset])
    series_res = pd.Series(average_result)
    series_vol = pd.Series(volatility)
    corr = series_res.corr(series_vol)
    res_on_vol = pearsonr(series_res, series_vol)
    t_test = ttest_ind(series_res, series_vol,equal_var=False)
    print(len(series_res))
    print(corr)
    print(res_on_vol)
    print(t_test)

def schichtgroeße_correlation():
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv",
        index_col=[0])
    result = []
    schicht = []
    for index, row in df.iterrows():
        result.append(row["average assets hold"])
        print()
        if "512" in row["iteration"]:
            schicht.append(512)
        else:
            schicht.append(256)
    print(result)
    print(schicht)
    series_res = pd.Series(result).dropna()
    series_vol = pd.Series(schicht).dropna()
    corr = series_res.corr(series_vol)
    res_on_vol = pearsonr(series_res, series_vol)
    print(corr)
    print(res_on_vol)


if __name__ == "__main__":
    correlation_matrix()