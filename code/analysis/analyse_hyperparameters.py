import pandas as pd

agent_list = ["q_learning_agent",
                      "actor_critic_agent",
                      # "turtle_agent",
                      # "moving_average_agent",
                      "duel_recurrent_q_learning_agent",
                      "DDPG"
                      #"improved_q_learning_agent"
                      ]
from collections import Counter

def collect_hyperparameter(agent):
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent + "_final_results.csv",
        index_col=[0])

    list_decay_rate, list_layer_size, list_memory_size, list_gamma = [],[],[],[]

    for index, row in df.iterrows():
        for fold in ["fold1", "fold2"]:
            string = row.loc["iteration"][8:]
            if len(string)>0:
                hyp_liste = string.split("_")
                decay_rate, layer_size, memory_size, gamma = hyp_liste[0], hyp_liste[1], hyp_liste[2], hyp_liste[3]
                list_decay_rate.append(decay_rate)
                list_layer_size.append(layer_size)
                list_memory_size.append(memory_size)
                list_gamma.append(gamma)

    count = {**Counter(list_decay_rate), **Counter(list_layer_size), **Counter(list_memory_size), **Counter(list_gamma)}

    try:
        result = pd.read_csv(
            "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent + "_hyperparameter.csv",
            index_col=[0])
    except FileNotFoundError:
        result = pd.DataFrame(columns=["value"])
    liste = ["decay_rate","decay_rate","layer_size","layer_size", "memory_size","memory_size","gamma","gamma"]
    i = 0
    for key, value in count.items():
        result.loc[liste[i]+"_"+str(key)] = value
        i += 1

    result.to_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/" + agent + "_hyperparameter.csv")


def hyperparameter_importance_val(parameter,value):
    df = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_hyperparameters.csv")
    result = []
    for fold in ["fold1", "fold2"]:
        for agent in agent_list:
            current_df = df.loc[df["agent"] == agent]
            current_df = df
            current_df = current_df.loc[current_df['iteration'].str.contains(fold)]
            for index, row in current_df.iterrows():
                string = row.loc["iteration"][8:]
                if len(string) > 0:
                    hyp_liste = string.split("_")
                    if len(hyp_liste)>4:
                        decay_rate, layer_size, memory_size, gamma, actor_lr,critic_lr,batch_size,tau  = \
                            hyp_liste[0], hyp_liste[1], hyp_liste[2], hyp_liste[3][:hyp_liste[3].rfind("9")+1], hyp_liste[3][hyp_liste[3].rfind("9")+1:], hyp_liste[4],hyp_liste[5],hyp_liste[6]
                        result.append([agent, row["total gains"], fold, decay_rate, layer_size, memory_size, gamma,actor_lr,critic_lr,batch_size,tau])
                    else:
                        decay_rate, layer_size, memory_size, gamma = hyp_liste[0], hyp_liste[1], hyp_liste[2], hyp_liste[3]
                        result.append([agent, row["total gains"], fold,decay_rate, layer_size, memory_size, gamma,0,0,0,0])

    result = pd.DataFrame(result, columns=["agent", "performance", "fold","decay_rate", "layer_size", "memory_size", "gamma","actor_lr","critic_lr","batch_size","tau"])
    for agent in agent_list:
        for fold in ["fold1", "fold2"]:
            print_res = result.loc[result["agent"] == agent]
            print_res = print_res.loc[print_res["fold"] == fold]
            print(agent, fold,print_res.loc[pd.to_numeric(print_res[parameter]) == value].performance.mean(),print_res.loc[pd.to_numeric(print_res[parameter]) != value].performance.mean())


def hyperparameter_importance_test(parameter,value):
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv")
    result = []
    df = df.fillna(0)
    for fold in ["fold1", "fold2"]:
        for agent in agent_list:
            current_df = df.loc[df["agent"] == agent]
            current_df = current_df.loc[current_df['iteration'].str.contains(fold)]
            for index, row in current_df.iterrows():
                string = row.loc["iteration"][8:]
                if len(string) > 0:
                    hyp_liste = string.split("_")
                    if len(hyp_liste)>4:
                        decay_rate, layer_size, memory_size, gamma, actor_lr,critic_lr,batch_size,tau  = \
                            hyp_liste[0], hyp_liste[1], hyp_liste[2], hyp_liste[3][:hyp_liste[3].rfind("9")+1], hyp_liste[3][hyp_liste[3].rfind("9")+1:], hyp_liste[4],hyp_liste[5],hyp_liste[6]
                        result.append([agent, row["total gains"], fold, decay_rate, layer_size, memory_size, gamma,actor_lr,critic_lr,batch_size,tau])
                    else:
                        decay_rate, layer_size, memory_size, gamma = hyp_liste[0], hyp_liste[1], hyp_liste[2], hyp_liste[3]
                        result.append([agent, row["total gains"], fold,decay_rate, layer_size, memory_size, gamma,0,0,0,0])

    result = pd.DataFrame(result, columns=["agent", "performance", "fold","decay_rate", "layer_size", "memory_size", "gamma","actor_lr","critic_lr","batch_size","tau"])
    for agent in agent_list:
        for fold in ["fold1", "fold2"]:
            print_res = result.loc[result["agent"] == agent]
            print_res = print_res.loc[print_res["fold"] == fold]
            print(agent, fold, print_res.loc[pd.to_numeric(print_res[parameter]) == value].performance.mean(),
                  print_res.loc[pd.to_numeric(print_res[parameter]) != value].performance.mean())

def count_hyperparameters(parameter,value,val):
    if val:
        df = pd.read_csv(
            "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_hyperparameters.csv")
    else:
        df = pd.read_csv(
            "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv")
    result = []
    for fold in ["fold1", "fold2"]:
        for agent in agent_list:
            current_df = df.loc[df["agent"] == agent]
            current_df = df
            current_df = current_df.loc[current_df['iteration'].str.contains(fold)]
            for index, row in current_df.iterrows():
                string = row.loc["iteration"][8:]
                if len(string) > 0:
                    hyp_liste = string.split("_")
                    if len(hyp_liste) > 4:
                        # 0.5_256_251_0.90.001_0.001_128_0.001
                        decay_rate, layer_size, memory_size, gamma, actor_lr, critic_lr, batch_size, tau = \
                            hyp_liste[0], hyp_liste[1], hyp_liste[2], hyp_liste[3][:hyp_liste[3].rfind("9") + 1], \
                            hyp_liste[3][hyp_liste[3].rfind("9") + 1:], hyp_liste[4], hyp_liste[5], hyp_liste[6]
                        result.append(
                            [agent, row["total gains"], fold, decay_rate, layer_size, memory_size, gamma, actor_lr,
                             critic_lr, batch_size, tau])
                    else:
                        decay_rate, layer_size, memory_size, gamma = hyp_liste[0], hyp_liste[1], hyp_liste[2], \
                                                                     hyp_liste[3]
                        result.append(
                            [agent, row["total gains"], fold, decay_rate, layer_size, memory_size, gamma, 0, 0, 0, 0])

    result = pd.DataFrame(result,
                          columns=["agent", "performance", "fold", "decay_rate", "layer_size", "memory_size", "gamma",
                                   "actor_lr", "critic_lr", "batch_size", "tau"])
    for agent in agent_list:
        for fold in ["fold1", "fold2"]:
            print_res = result.loc[result["agent"] == agent]
            print_res = print_res.loc[print_res["fold"] == fold]
            print(agent, fold, print_res.loc[pd.to_numeric(print_res[parameter]) == value].performance.count(),
                  print_res.loc[pd.to_numeric(print_res[parameter]) != value].performance.count())

if __name__ == "__main__":
    param_grid = {
        'decay_rate': [0.01, 0.001],
        'layer_size': [256, 512],
        'memory_size': [251, 753],
        'gamma': [0.9, 0.999]
    }
    hyperparameter_importance_val("tau",0.5)
    hyperparameter_importance_test("tau",0.5)