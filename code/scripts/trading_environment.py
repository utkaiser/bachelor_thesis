import os
import sys
import logging
import yfinance as yf
import pandas as pd
import tensorflow as tf
import pathlib

class Trading_env():

    '''
        Goal: create a realistic testbench containing all relevant assets, settings and parameters for the evaluation
        Author: Luis Kaiser, University of Wuerzburg, Bachelor Thesis
    '''

    def __init__(self, evaluation_name, relevant_assets, file, starting_date='2010-01-02', ending_date='2019-12-31'):

        '''
            Goal: create testbench with all assets, settings and parameters

            Input:
            - evaluation_name: Name of the evaluation
            - file: file which is in the working directory (main-file)
            - relevant_assets: relevant asset, etf or commodity to trade with
            - iterations: parameter for training

            Output:/
        '''

        # global fix parameters agent
        self.initial_money = 100000  #chosen similar to related work
        self.checkpoint = 10 #for documentation of process
        self.iterations = 1000
        self.output_size = 3 #predefined
        self.action_size = self.output_size
        self.min_epsilon = 0 #set to 0 to reduce number of hyperparameters, therefore, epsilon is only defined by decay rate

        self.state_size = 30

        # data
        self.starting_date = starting_date
        self.ending_date = ending_date
        self.transaction_cost = 0.01  # in percent

        # settings
        tf.compat.v1.reset_default_graph()
        self.workdir_settings(file)
        self.relevant_asset = relevant_assets
        self.result_dir = os.getcwd() + "/results/" + evaluation_name
        self.create_folder_with_log()
        self.number_runs = 3
        self.require_improvement = 15 #for early stopping, similar to other papers (Deep RL for Trading Zihao zhang, ...)

        self.assets = {

            ###############################################################################
            ###################################### NYSE ###################################
            ###############################################################################

            #US companies
            #technology
            #big
            'Apple': 'AAPL',
            'Oracle Corporation': 'ORCL',
            'Accenture plc': 'ACN',
            'Microsoft Corporation':'MSFT',
            'International Business Machines Corporation':'IBM',
            'Cisco Systems, Inc.':'CSCO',
            'Nvidia Corporation':'NVDA',
            'Adobe Inc.':'ADBE',
            'HP Inc.': 'HPQ',
            'Intel Corporation': 'INTC',
            #small
            'TESSCO Technologies': 'TESS',
            'Amtech Systems': 'ASYS',
            'Computer Task Group': 'CTG',
            'Bel Fuse':'BELFB',
            'Aviat Networks':'AVNW',
            'LSI Industries':'LYTS',

            #financials
            'JPMorgan Chase': 'JPM',
            'Bank of America Corporation': 'BAC',
            'Visa Inc.': 'V',

            #health care
            'Pfizer Inc.': 'PFE',
            'Merck & Co., Inc.': 'MRK',
            'Johnson & Johnson': 'JNJ',

            #
            #Asian companies

            #technology
            #big
            'Canon Inc.':'CAJ',
            'NICE Ltd.':'NICE',
            'Taiwan Semiconductor Manufacturing Company Limited':'TSM',
            'Sony Corporation':'SNE',
            'United Microelectronics Corporation':'UMC',
            'Check Point Software Technologies Ltd.': 'CHKP',
            #small
            'Silicom Ltd.': 'SILC',
            'Gilat Satellite Networks Ltd.': 'GILT',
            'Tower Semiconductor Ltd.': 'TSEM',

            #financials
            'China Life Insurance Company Limited': 'LFC',
            'Sumitomo Mitsui Financial Group, Inc.': 'SMFG',
            'Shinhan Financial Group Co., Ltd.': 'SHG',


            #European companies

            #technology
             'Nokia Corporation':'NOK',
             'ASML Holding N.V.':'ASML',
             'Telefonaktiebolaget LM Ericsson':'ERIC',
             'SAP SE':'SAP',
             'TE Connectivity Ltd.':'TEL',
             'Logitech International S.A.':'LOGI',

            #financials
             'HSBC Holdings plc':'HSBC',
             'ING Groep N.V.':'ING',
             'Barclays PLC':'BCS',


            ###############################################################################
            ###################################### HKSE ###################################
            ###############################################################################

            #technology
            'Lenovo Group Limited':'0992.HK',
            'Kingsoft Corporation Ltd':'3888.HK',
            'ZTE Corporation':'0763.HK',

            #financials
            'China Construction Bank Corporation':'0939.HK',
            'Ping An Insurance(Group) Company of China, Ltd.':'2318.HK',
            'China CITIC Bank Corporation Limited':'0998.HK',

            ###############################################################################
            ###################################### COMEX ##################################
            ###############################################################################

            #metal
            'Gold':'GC=F',
            'Silver':'SI=F',
            'Platinum':'PL=F',
            'Copper': 'HG=F',

            #energy
            'Crude Oil':'CL=F',
            'Heating Oil':'HO=F',
            'Natural Gas':'NG=F',
            'RBOB Gasoline':'RB=F',

            #livestock and meat
            'Lean Hogs Futures':'HE=F',
            'Live Cattle Futures':'LE=F',
            'Feeder Cattle Futures':'GF=F',
            'Class III Milk Futures':'DC=F',

            #agriculture
            'Corn Futures':'ZC=F',
            'Soybean Futures':'ZS=F',
            'Coffee':'KC=F',
            'KC HRW Wheat Futures':'KE=F',

            ###############################################################################
            ###################################### ETF ####################################
            ###############################################################################

            #general
            #USA
            'Dow Jones':'DIA',
            'S&P 500':'SPY',
            'NASDAQ':'QQQ',

            #Asia
            'Nikkei 225':'EWJ',
            'iShares MSCI Taiwan ETF':'EWT',
            'iShares MSCI South Korea ETF':'EWY',

            #Europe
            'FTSE 100':'EZU',
            'Lyxor CAC 40(DR) UCITS ETF Dist':'CAC.PA',
            'iShares Core DAX UCITS ETF(DE)':'EXS1.DE',

            #commodities
            'iShares Diversified Commodity Swap UCITS ETF(DE)':'EXXY.MI',
            'Invesco DB Commodity Index Tracking Fund':'DBC'
        }

        self.relevant_data = self.preprocess_data_from_yfinance(relevant_assets)
        logging.info("Evaluate agent on " + self.relevant_asset + ".")



    def create_folder_with_log(self):

        '''
            Goal: creates folder under dirname, raises Error if directory already exists, prints parameters used
            Input: /
            Output:
            - list of used parameters in logfile
            - directories to store results and plots
            - creates logfile in which programm documents process
        '''
        if not os.path.isdir(self.result_dir[:self.result_dir.rfind("/")]):
            os.mkdir(self.result_dir[:self.result_dir.rfind("/")])

        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        i = 1
        while i > 0:
            p = pathlib.Path(self.result_dir + "/logfile" + str(i) + ".log")
            if not p.is_file():
                break
            i += 1

        logging.basicConfig(filename=self.result_dir+ "/logfile" + str(i) + ".log",
                            level=logging.INFO,
                            format='[%(asctime)s] %(filename)s:%(lineno)d: %(message)s (%(levelname)s)',
                            datefmt='%H:%M:%S')

    def workdir_settings(self, file):

        '''
            Goal: defines working directory, needed for Docker in LSX-Cluster
            Input: /
            Output: /
        '''

        work_dir_path = os.path.dirname(file)
        os.chdir(work_dir_path)  # set working directory
        sys.path.append(work_dir_path)


    def preprocess_data_from_yfinance(self, relevant_assets):

        '''
            Goal: download and preprocess data from yfinance

            Input:
            - relevant assets to trade with

            Output:
            - dataframe with relevant assets
        '''

        result = []
        for asset_name, asset_ticker in self.assets.items():
            if asset_ticker in relevant_assets:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=self.starting_date, end=self.ending_date)["Close"]
                series.name = asset_ticker
                result.append(series)
        df = pd.concat(result, axis=1)
        return df

    def preprocess_context_data(self,asset_name,corona = False):
        starting_date = '2010-01-02'
        if corona:
            ending_date = '2020-12-31'
        else:
            ending_date = '2019-12-31'
        result = []
        if asset_name in ["ORCL","AAPL","ACN","MSFT","IBM","CSCO","NVDA","ADBE","HPQ","INTC","TESS","ASYS","CTG","BELFB","AVNW","LYTS"]:
            for asset_ticker in ["GC=F","^TNX","SPY","AVGO","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["JPM","BAC","V"]:
            for asset_ticker in ["GC=F","^TNX","SPY","BLK","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["PFE","MRK","JNJ"]:
            for asset_ticker in ["GC=F","^TNX","SPY","AZN","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["CAJ","NICE","TSM","SNE","UMC","CHKP","SILC","GILT","TSEM"]:
            for asset_ticker in ["GC=F","^TNX","SPY","CHA","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["LFC","SMFG","SHG"]:
            for asset_ticker in ["GC=F","^TNX","SPY","NMR","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["NOK","ASML","ERIC","SAP","TEL","LOGI"]:
            for asset_ticker in ["GC=F","^TNX","SPY","STM","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["HSBC","ING","BCS"]:
            for asset_ticker in ["GC=F","^TNX","SPY","CS","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["0992.HK","3888.HK","0763.HK"]:
            for asset_ticker in ["GC=F","^TNX","EWJ","0700.HK","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["0939.HK","2318.HK","0998.HK"]:
            for asset_ticker in ["GC=F","^TNX","SPY","3968.HK","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["GC=F","SI=F","PL=F","HG=F"]:
            for asset_ticker in ["GC=F","^TNX","SPY","ALI=F","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["CL=F","HO=F","NG=F","RB=F"]:
            for asset_ticker in ["GC=F","^TNX","SPY","BZ=F","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["HE=F","LE=F","GF=F"]:
            for asset_ticker in ["GC=F","^TNX","SPY","DC=F","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["DC=F"]:
            for asset_ticker in ["GC=F","^TNX","SPY","LE=F","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["ZC=F","ZS=F","KC=F","KE=F"]:
            for asset_ticker in ["GC=F","^TNX","SPY","CC=F","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["DIA","SPY","QQQ"]:
            for asset_ticker in ["GC=F","^TNX","SPY","^NYA","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["EWJ","EWT","EWY"]:
            for asset_ticker in ["GC=F","^TNX","SPY","^HSI","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["EZU","CAC.PA","EXS1.DE"]:
            for asset_ticker in ["GC=F","^TNX","SPY","^IBEX","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        elif asset_name in ["EXXY.MI","DBC"]:
            for asset_ticker in ["GC=F","^TNX","SPY","GSG","^VIX"]:
                asset = yf.Ticker(asset_ticker)
                series = asset.history(start=starting_date, end=ending_date)["Close"]
                asset = yf.Ticker(asset_name)
                my_asset = asset.history(start=starting_date, end=ending_date)["Close"]
                series = series[series.index.isin(my_asset.index)]
                series.name = asset_ticker
                result.append(series)
        df = pd.concat(result, axis=1)
        df = df.fillna(method='ffill')
        return df

