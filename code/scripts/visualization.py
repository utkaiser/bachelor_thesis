import matplotlib.pyplot as plt
import datetime

def create_plot_after_preprocessing(series,column,markers):

    '''
        Goal: visualizes whole time series after preprocessing, saves it in file

        Input:
        - series: df with all assets
        - column: name of asset to trade with
        - markers: timestamps to display the folds with vertical lines in plot

        Output:
        - data visualization with correct title, axis, format and markers which display the folds
    '''

    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    ax.set_facecolor('w')
    plt.plot(series, color='blue', lw=1.)
    plt.grid(False)
    plt.suptitle(column, fontsize=20)
    plt.title("time series to train and test the agents on", fontsize=15, y=1)
    plt.xlabel("Date")
    plt.ylabel("Close value")
    fig.autofmt_xdate()
    if isinstance(markers, datetime.date):
        plt.axvline(x=markers)
    else:
        for marker in markers:
            plt.axvline(x=marker)
    result = fig
    plt.close()
    return result


def create_plot_result(series, title, states_buy, states_sell, total_gains, invest):

    '''
        Goal: visualizes result plot in file with all other results of agenten trading with one asset

        Input:
        - series: time series of asset
        - title: title of this plot, usually: <<evaluation_name>>_<<ticker of asset>>
        - states_buy: timestamps Agent bought asset
        - states_sell: timestamp Agent sold asset
        - total_gains: absolute value of gained money in test
        - invest: percentage of initial money gained in test

        Output:
        - data visualization which displays trading behaviour of Agent
    '''

    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    ax.set_facecolor('w')
    plt.plot(series, color='blue', lw=2.)
    plt.grid(False)
    if len(states_buy) > 0:
        plt.plot(series, '^', markersize=10, color='g', label='buying signal', markevery=states_buy)
    if len(states_sell) > 0:
        plt.plot(series, 'v', markersize=10, color='r', label='selling signal', markevery=states_sell)
    plt.suptitle(title, fontsize=20)
    plt.title('total gains %f, total investment %f%%' % (total_gains, invest), fontsize=15, y=1)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Close value")
    fig.autofmt_xdate()
    result = fig
    plt.close()
    return result

def create_plot_portfolio_movement(df, stock):
    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    ax.set_facecolor('w')
    plt.plot(df, lw=1.)
    plt.grid(False)
    plt.title(stock, fontsize=20)
    plt.xlabel("date")
    plt.ylabel("portfolio value")
    ax.legend()
    fig.autofmt_xdate()
    plt.legend(df.columns)
    result = fig
    plt.close()
    return result

def create_plot_after_preprocessing_repo2(series,column,markers):
    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    ax.set_facecolor('w')
    plt.plot(series, color='blue', lw=1.)
    plt.grid(False)
    plt.suptitle(column, fontsize=20)
    plt.title("time series to train and test the agents on", fontsize=15, y=1)
    plt.xlabel("Date")
    plt.ylabel("Close value")
    fig.autofmt_xdate()
    if isinstance(markers, datetime.date):
        plt.axvline(x=markers)
    else:
        for marker in markers:
            plt.axvline(x=marker)
    result = fig
    plt.close()
    return result
