import pandas as pd

agent_list = ["q_learning_agent",
                      "actor_critic_agent",
                      "turtle_agent",
                      "moving_average_agent",
                      "duel_recurrent_q_learning_agent",
                      #"ddpg_agent"
                      #"improved_q_learning_agent"
                      ]

def get_best_agent_performance(metric):
    existing_df = pd.DataFrame(columns=["iteration","total gains","volatility","greatest loss",
                             "sharpe ratio","agent","asset","run","class","exchange","region","sector","company size","volatility asset","beta","course development"])
    for fold in ["fold1","fold2"]:
        for agent in agent_list:
            df = pd.read_csv(
                "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent + "_performance_analysis.csv",
                index_col=[0])
            result = {}
            df = df.loc[df['iteration'].str.contains(fold)]
            df = df.loc[df[metric] == df[metric].max()]
            row = df.values.tolist()[0]
            result[agent] = row
            result = pd.DataFrame.from_dict(result, orient='index', columns=existing_df.columns.values.tolist())
            existing_df = existing_df.append(result)
    existing_df.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/best_" + metric.replace(" ","_") + ".csv")

def average_performance():
    result = pd.DataFrame()
    for agent in agent_list:
        df = pd.read_csv(
            "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent + "_performance_analysis.csv",
            index_col=[0])
        df["agent"] = agent
        result = result.append(df)
    result.index = range(len(result))
    result.to_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv")

    result = result.loc[result['iteration'].str.contains("fold1")]
    print(result["total gains"].mean())

def clustering_performance(aspect,property,metric):
    for fold in ["fold1"]:
        df = pd.read_csv(
            "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results_corona.csv",
            index_col=[0])
        df = df.loc[df['iteration'].str.contains(fold)]
        print(df['asset'].unique())
        df_current = df
        #df = df.loc[df["agent"] == agent]
        if aspect == "beta" or aspect == "volatility asset" or aspect == "mean difference asset":
            df_current = df_current.loc[df_current[aspect] > property]
            df = df.loc[df[aspect] <= property]
        else:
            df_current = df_current.loc[df_current[aspect] == property]
            df = df.loc[df[aspect] != property]

        print(fold, "mean difference asset", df_current["mean difference asset"].mean(), df["mean difference asset"].mean())
        print(fold, "beta", df_current["beta"].mean(), df["beta"].mean())
        print(fold, "volatility mean", df_current["volatility asset"].unique().mean(),df["volatility asset"].unique().mean())
        print(fold, "total mean" ,df_current[metric].unique().mean(),df[metric].unique().mean())

def compare_single_stocks(asset1,asset2,metric):
    for fold in ["fold1","fold2"]:
        df = pd.read_csv(
            "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv",
            index_col=[0])
        df = df.loc[df['iteration'].str.contains(fold)]
        df_1 = df.loc[df["asset"] == asset1]
        df_2 = df.loc[df["asset"] == asset2]
        print(fold,df_1[metric].mean())
        print(fold, df_2[metric].mean())

def count_loss_and_profit():
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv",
        index_col=[0])
    for agent in agent_list:
        for fold in ["fold1","fold2"]:
            current_df = df
            current_df = current_df.loc[current_df['iteration'].str.contains(fold)]
            print(agent,len(current_df.loc[current_df["total gains"] >= 0]), len(current_df.loc[current_df["total gains"] < 0]))

def average_performance():
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv",
        index_col=[0])
    print(df[df["iteration"].str.contains("fold1")]["total gains"].mean())
    print(df[df["iteration"].str.contains("fold2")]["total gains"].mean())

def first_look():
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/improved_q_learning_agent_final_results.csv",
        index_col=[0])
    print(df[df["iteration"].str.contains("0.5")]["total gains"].mean())

if __name__ == "__main__":
    first_look()