import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.dates as mdates

from matplotlib.backends.backend_pdf import PdfPages


# import matplotlib
# pgf_with_latex = {                      # setup matplotlib to use latex for output
#     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#     "text.usetex": True,                # use LaTeX to write all text
#     "font.family": "serif",
#     "font.serif": [],                   # blank entries should cause plots
#     "font.sans-serif": [],              # to inherit fonts from the document
#     "font.monospace": [],
#     "axes.labelsize": 10,
#     "font.size": 10,
#     "legend.fontsize": 8,               # Make the legend/label fonts
#     "xtick.labelsize": 8,               # a little smaller
#     "ytick.labelsize": 8,
#     "figure.figsize": [8.27, 5.845],     # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[utf8]{inputenc}",    # use utf8 input and T1 fonts
#         r"\usepackage[T1]{fontenc}",        # plots will be generated
#         ]                                   # using this preamble
#     }
# matplotlib.use("pgf")
# matplotlib.rcParams.update(pgf_with_latex)

def plot_portfolio_movement():
    df = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/q_learning_agent/AAPL/run2/portfolio_movements_1.csv",index_col=[0])
    result = pd.DataFrame()
    result["AAPL"] = df["fold2___0.01_256_753_0.999"]
    df_2 = pd.read_csv("/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/q_learning_agent/V/run3/portfolio_movements_1.csv",index_col=[0])
    print(df_2)
    result["V"] = df_2["fold2___0.01_256_753_0.999"]

    fig = plt.figure(figsize=(20, 15))
    ax = plt.gca()
    ax.set_facecolor('w')
    plt.plot(result, lw=1.)
    plt.grid(False)
    plt.title("Vergleich", fontsize=20)
    plt.xlabel("date")
    plt.ylabel("portfolio value")
    ax.legend()
    fig.autofmt_xdate()
    plt.legend(result.columns)
    result = fig
    plt.show()

def boxplot():
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results.csv",
        index_col=[0])
    df_corona = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_results_corona.csv",
        index_col=[0])
    fig, ax = plt.subplots(2,3,figsize=(7.5, 5.845),gridspec_kw = {'wspace':0.22, 'hspace':0.14}, dpi = 3500)
    medianprops = {'color': 'red', 'markersize':2.5}
    boxprops = {'color': 'black','linewidth':0.5}
    whiskerprops = dict(linestyle='--', color='black', linewidth=0.5)
    meanprops = {'marker':'s','color': 'red','markerfacecolor':'red','markeredgecolor':'black',"markersize":2.5}
    pdf = PdfPages("/Users/udis/Downloads/plot.pdf")
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Times News Roman'
    for j in range(0,2):
        if j == 1:
            df = df.loc[df["class"] == "stock"]
            df_corona = df_corona.loc[df_corona["class"] == "stock"]
        i = 0
        for fold in ["fold1","fold2","fold3"]:
            metric = "total gains"
            if fold != "fold3":
                df_current = df.loc[df['iteration'].str.contains(fold)]
                #metric = "sharpe ratio"
                df_1 = df_current[df_current["agent"] == "q_learning_agent"][metric]
                df_2 = df_current[df_current["agent"] == "actor_critic_agent"][metric]
                df_3 = df_current[df_current["agent"] == "duel_recurrent_q_learning_agent"][metric]
                df_6 = df_current[df_current["agent"] == "moving_average_agent"][metric]
                df_7 = df_current[df_current["agent"] == "turtle_agent"][metric]
            else:
                df_current = df_corona
                # metric = "sharpe ratio"
                df_1 = df_current[df_current["agent"] == "q_learning_agent"][metric]
                df_2 = df_current[df_current["agent"] == "actor_critic_agent"][metric]
                df_3 = df_current[df_current["agent"] == "duel_recurrent_q_learning_agent"][metric]
                df_6 = df_current[df_current["agent"] == "moving_average_agent"][metric]
                df_7 = df_current[df_current["agent"] == "turtle_agent"][metric]



            data = [df_1,df_2,df_3,df_6,df_7]
            ax[j][i].boxplot(data,
                             showmeans=True,
                             showfliers=False,
                             whis=1.5,
                             widths=0.42,
                             meanprops=meanprops,
                             medianprops=medianprops,
                             whiskerprops = whiskerprops,
                             boxprops=boxprops,
                             capprops=dict(linestyle='-', linewidth=0.5, color='Black')
                             )
            ax[j][i].spines['top'].set_visible(False)

            ax[j][i].set_xticklabels(["DQLA", "DDPGA", "DDRQLA", "MA", "TT"], font="Times", fontsize=6)

            if j == 0:
                ax[j][i].set_ylim(-850, 1200)
                if i == 0:
                    ax[j][i].set_title('2018 (alle Wertpapiere)',font="Times",size=8,pad=-1, fontweight="bold")
                elif i == 1:
                    ax[j][i].set_title('2019 (alle Wertpapiere)',font="Times",size=8,pad=-1)
                else:
                    ax[j][i].set_title('2020 (alle Wertpapiere)',font="Times",size=8,pad=-1)
            else:
                ax[j][i].set_ylim(-1100, 1200)
                if i == 0:
                    ax[j][i].set_title('2018 (nur Aktien)',font="Times",size=8,pad=-100)
                elif i == 1:
                    ax[j][i].set_title('2019 (nur Aktien)',font="Times",size=8,pad=-100)
                else:
                    ax[j][i].set_title('2020 (nur Aktien)',font="Times",size=8,pad=-100)

            ax[j][i].set_ylabel("Profit in Dollar",labelpad=-1,font="Times", fontsize=5)
            ax[j][i].yaxis.set_tick_params(labelsize=5,pad=1.1)
            ax[j][i].xaxis.set_tick_params(pad=1.4)
            ax[j][i].tick_params(width=0.5, direction="in", length=1.5)
            plt.setp(ax[j][i].spines.values(), linewidth=0.5)
            i += 1


    plt.tight_layout()
    plt.grid(False)
    result = fig
    #plt.close()

    #plt.show()
    #pdf.savefig(result)
    plt.savefig("/Users/udis/Downloads/plot.pdf", dpi=3500)
    #tikzplotlib.save("/Users/udis/Downloads/test.tex")

def standard_deviation_plot():
    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_portfolio_movements.csv",
        index_col=[0])

    selected_columns_11,selected_columns_12,selected_columns_13 = [],[],[]
    selected_columns_21, selected_columns_22, selected_columns_23 = [], [], []
    selected_columns_31, selected_columns_32, selected_columns_33 = [], [], []
    for entry in df.columns.values.tolist():
        if "agent" in entry:
            if "fold1" in entry:
                if entry.endswith("1"):
                    selected_columns_11.append(entry)
                elif entry.endswith("2"):
                    selected_columns_12.append(entry)
                else:
                    selected_columns_13.append(entry)
            elif "fold2" in entry:
                if entry.endswith("1"):
                    selected_columns_21.append(entry)
                elif entry.endswith("2"):
                    selected_columns_22.append(entry)
                else:
                    selected_columns_23.append(entry)
            elif "fold2" in entry:
                if entry.endswith("1"):
                    selected_columns_31.append(entry)
                elif entry.endswith("2"):
                    selected_columns_32.append(entry)
                else:
                    selected_columns_33.append(entry)
    selected_columns_31, selected_columns_32, selected_columns_33 = selected_columns_21, selected_columns_22, selected_columns_23
    to_mean_list = [[selected_columns_11,selected_columns_12,selected_columns_13],
                    [selected_columns_21, selected_columns_22, selected_columns_23],
                    [selected_columns_31, selected_columns_32, selected_columns_33]]

    average_market_df = market_average_quantile()

    i = 0
    fig, ax = plt.subplots(3, 3,figsize=(20,15))
    for selected_columns_1, selected_columns_2, selected_columns_3 in to_mean_list:
        #xaxis --> fold, yaxis-->quantil beta
        df_1 = df[selected_columns_1]
        df_2 = df[selected_columns_2]
        df_3 = df[selected_columns_3]
        df_1['mean'] = df_1.mean(axis=1)
        df_2['mean'] = df_2.mean(axis=1)
        df_3['mean'] = df_3.mean(axis=1)
        ax[0,i].plot(df_1['mean'].index,df_1['mean'].values/100000, color='blue')  # meanprops=, medianprops=, boxprops=
        ax[0,i].fill_between(df_1['mean'].index,df_1['mean'].values/100000 - (df_1['mean']/100000).std(),
                             df_1['mean'].values/100000+(df_1['mean']/100000).std(), facecolor='lightsteelblue')
        ax[0,i].set_ylim([.99800, 1.00100])
        ax[1,i].plot(df_2['mean'].index, df_2['mean'].values/100000, color='blue')
        ax[1, i].fill_between(df_2['mean'].index, df_2['mean'].values/100000 - (df_2['mean']/100000).std(),
                              df_2['mean'].values/100000 + (df_2['mean'].values/100000).std(), facecolor='lightsteelblue')
        ax[1, i].set_ylim([.99800, 1.00300])
        ax[2,i].plot(df_3['mean'].index, df_3['mean'].values/100000, color='blue')
        ax[2, i].fill_between(df_3['mean'].index, df_3['mean'].values/100000 - (df_3['mean']/100000).std(),
                              df_3['mean'].values/100000 + (df_3['mean'].values/100000).std(), facecolor='lightsteelblue')
        ax[2, i].set_ylim([.99000, 1.00700])

        ax21 = ax[0, i].twinx()
        ax22 = ax[1, i].twinx()
        ax23 = ax[2, i].twinx()
        ax21.plot(average_market_df[average_market_df.columns[i]].index,
                 average_market_df[average_market_df.columns[i]].values, color='green',linestyle="--",dashes=(5, 3))

        ax22.plot(average_market_df[average_market_df.columns[i+3]].index,
                 average_market_df[average_market_df.columns[i+3]].values, color='green', linestyle="--",dashes=(5, 3))
        ax23.plot(average_market_df[average_market_df.columns[i+6]].index,
                 average_market_df[average_market_df.columns[i+6]].values, color='green', linestyle="--",dashes=(5, 3))
        ax21.set_ylim([0.5, 1.5])
        ax22.set_ylim([0.5, 1.5])
        ax23.set_ylim([0.5, 1.5])

        ax21.tick_params(axis='y', colors='green')
        ax22.tick_params(axis='y', colors='green')
        ax23.tick_params(axis='y', colors='green')

        i += 1

    ax[2, 0].set(xlabel="Handelstage 2018")
    ax[2, 1].set(xlabel="Handelstage 2019")
    ax[2,2].set(xlabel = "Handelstage 2020")
    ax[0, 0].set(ylabel="relativer Portfoliowert (erstes Quantil)")
    ax[1, 0].set(ylabel="relativer Portfoliowert (zweites Quantil)")
    ax[2, 0].set(ylabel="relativer Portfoliowert (drittes Quantil)")

    for axx in ax.flat:
        axx.label_outer()
    plt.grid(False)
    plt.show()

def scale(x):
    return x /100000

def scale_2(x):
    return x *100000

def market_average_quantile():
    starting_date = '2010-01-02'
    ending_date = '2019-12-31'

    df = pd.read_csv(
        "/Users/udis/Desktop/Bachelorarbeit/GitLab/bachelor_thesis/results/repository1/collected_results/all_portfolio_movements.csv",
        index_col=[0])
    result = []
    column_list = df.columns.values.tolist()
    already_in = []
    for column in column_list:
        asset_name = column[column.index('agent')+6:column.index('run')-1]
        if asset_name not in already_in:
            asset = yf.Ticker(asset_name)
            series = asset.history(start=starting_date, end=ending_date)["Close"]
            series.name = asset_name+"_"+column[-1]
            result.append(series)
            already_in.append(asset_name)
    result = pd.concat(result, axis=1)
    result = result.fillna(method='ffill')
    cols_1 = [col for col in result.columns.values.tolist() if col.endswith('1')]
    cols_2 = [col for col in result.columns.values.tolist() if col.endswith('2')]
    cols_3 = [col for col in result.columns.values.tolist() if col.endswith('3')]
    result_1 = result[cols_1]
    result_1['mean'] = result_1.mean(axis=1)
    result_2 = result[cols_2]
    result_2['mean'] = result_2.mean(axis=1)
    result_3 = result[cols_3]
    result_3['mean'] = result_3.mean(axis=1)
    final_result = pd.DataFrame()
    final_result["quantil1_2018"] = result_1['mean'].div(result_1['mean'][-502:-251].iloc[0])[-502:-251].reset_index(drop=True)
    final_result["quantil1_2019"] = result_1['mean'].div(result_1['mean'][-251:].iloc[0])[-251:].reset_index(drop=True)
    final_result["quantil1_2020"] = result_1['mean'].div(result_1['mean'][-251:].iloc[0])[-251:].reset_index(drop=True)
    final_result["quantil2_2018"] = result_2['mean'].div(result_2['mean'][-502:-251].iloc[0])[-502:-251].reset_index(drop=True)
    final_result["quantil2_2019"] = result_2['mean'].div(result_2['mean'][-251:].iloc[0])[-251:].reset_index(drop=True)
    final_result["quantil2_2020"] = result_2['mean'].div(result_2['mean'][-251:].iloc[0])[-251:].reset_index(drop=True)
    final_result["quantil3_2018"] = result_3['mean'].div(result_3['mean'][-502:-251].iloc[0])[-502:-251].reset_index(drop=True)
    final_result["quantil3_2019"] = result_3['mean'].div(result_3['mean'][-251:].iloc[0])[-251:].reset_index(drop=True)
    final_result["quantil3_2020"] = result_3['mean'].div(result_3['mean'][-251:].iloc[0])[-251:].reset_index(drop=True)
    final_result = final_result.reset_index(drop=True)
    return final_result

def states_buy():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Times'
    fig, ax = plt.subplots(2, figsize=(7.5, 5),gridspec_kw = {'hspace':0.3}, dpi = 3500)
    asset = yf.Ticker("DIA")
    series_asset = asset.history(start='2020-01-01', end='2020-12-31')["Close"]
    print(series_asset.index.tolist())
    for i in range(2):
        ax[i].plot(series_asset, color='black', lw=.5)
        ax[i].spines['top'].set_visible(False)
        ax[i].set_ylabel("tägliche Schlusskurse in Dollar", labelpad=3, font="Times", fontsize=6)
        ax[i].yaxis.set_tick_params(labelsize=6, pad=1.2)

        ax[i].tick_params(width=0.5, direction="in", length=1.5)
        plt.setp(ax[i].spines.values(), linewidth=0.5)
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax[i].xaxis.set_tick_params(labelsize=7, pad=1.8)
        if i == 0:
            ax[i].set_title('deep q learning Agent', font="Times", size=9, pad=-1)
            ax[i].plot(series_asset, '^', markersize=2, color='g', label='Kaufsignal', markevery=buy_list_q)
            ax[i].plot(series_asset, 'v', markersize=2, color='r', label='Verkaufsignal', markevery=sell_list_q)
        elif i == 1:
            ax[i].set_title('deep recurrent q learning Agent', font="Times", size=9, pad=-1)
            ax[i].plot(series_asset, '^', markersize=2, color='g', label='Kaufsignal', markevery=buy_list_duel)
            ax[i].plot(series_asset, 'v', markersize=2, color='r', label='Verkaufsignal', markevery=sell_list_duel)

    plt.tight_layout()
    plt.grid(False)
    plt.savefig("/Users/udis/Downloads/plot2.pdf", dpi=3500)


if __name__ == "__main__":
    states_buy()









