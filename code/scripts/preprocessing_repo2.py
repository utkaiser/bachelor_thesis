import pandas as pd

def preprocess_repo2(df,asset,Trading_env):
    df = df.to_frame()

    index = df.index.strftime('%Y-%m-%d')
    index = [str.replace("-", "") for str in index]
    liste = []
    for i in index:
        liste.append(i)
        liste.append(i)

    df_context = Trading_env.preprocess_context_data(asset)
    df['gold'] = df_context.iloc[:,0]
    df['interest'] = df_context.iloc[:,1]
    df['index'] = df_context.iloc[:,2]
    df['similar'] = df_context.iloc[:,3]
    df['vix'] = df_context.iloc[:,4]

    columns = df.columns.values.tolist()
    columns[0] = 'adjcp'
    df.columns = columns

    #df = df.rename(columns={'AAPL': 'adjcp'})
    df = df.reset_index(drop=True)
    df.index = range(0, len(df) * 2, 2)
    for i in range(1, len(df) * 2, 2):
        line = pd.DataFrame({"gold": 1, "interest": 1, "index": 1, "similar": 1, 'adjcp': 1, 'vix':1}, index=[i])
        df = df.append(line, ignore_index=False)

    df = df.sort_index().reset_index(drop=True)
    df["datadate"] = liste
    liste = []
    for i in range(0, len(df) // 2):
        liste.append(i)
        liste.append(i)
    df.index = liste
    df["datadate"] = pd.to_numeric(df["datadate"])

    fold1 = df[(df.datadate > 20100103) & (df.datadate <= 20161231)]
    fold2 = df[(df.datadate > 20170101) & (df.datadate <= 20171231)]
    fold3 = df[(df.datadate > 20180101) & (df.datadate <= 20181231)]
    fold4 = df[(df.datadate > 20190101) & (df.datadate <= 20191231)]
    ind1, ind2, ind3 = [], [], []
    longerfold = fold1.append(fold2)
    for i in range(0, len(fold1) // 2):
        ind1.append(i)
        ind1.append(i)
    for i in range(0, len(fold2) // 2):
        ind2.append(i)
        ind2.append(i)
    for i in range(0, len(longerfold) // 2):
        ind3.append(i)
        ind3.append(i)

    fold1.index = ind1
    fold2.index = ind2
    try:
        fold3.index = ind2[:len(fold3.index)]
        fold4.index = ind2[:len(fold4.index)]
    except ValueError:
        fold3.index = ind2[:len(fold3.index)]+[len(fold2) // 2 +2,len(fold2) // 2+2,len(fold2) // 2 +3,len(fold2) // 2+3]
    longerfold.index = ind3

    return [[fold1, fold2, fold3], [longerfold, fold3, fold4]]

def merge_folds_test(fold1,fold2):
    longerfold = fold1.append(fold2)
    ind3 = []
    for i in range(0, len(longerfold) // 2):
        ind3.append(i)
        ind3.append(i)
    longerfold.index = ind3
    return longerfold

def preprocess_repo2_corona(df,asset,Trading_env):
    df = df.to_frame()
    index = df.index.strftime('%Y-%m-%d')
    index = [str.replace("-", "") for str in index]
    liste = []
    for i in index:
        liste.append(i)
        liste.append(i)

    df_context = Trading_env.preprocess_context_data(asset, True)
    df['gold'] = df_context.iloc[:,0]
    df['interest'] = df_context.iloc[:,1]
    df['index'] = df_context.iloc[:,2]
    df['similar'] = df_context.iloc[:,3]
    df['vix'] = df_context.iloc[:,4]

    columns = df.columns.values.tolist()
    columns[0] = 'adjcp'
    df.columns = columns

    #df = df.rename(columns={'AAPL': 'adjcp'})
    df = df.reset_index(drop=True)
    df.index = range(0, len(df) * 2, 2)
    for i in range(1, len(df) * 2, 2):
        line = pd.DataFrame({"gold": 1, "interest": 1, "index": 1, "similar": 1, 'adjcp': 1, 'vix':1}, index=[i])
        df = df.append(line, ignore_index=False)

    df = df.sort_index().reset_index(drop=True)
    df["datadate"] = liste
    liste = []
    for i in range(0, len(df) // 2):
        liste.append(i)
        liste.append(i)
    df.index = liste
    df["datadate"] = pd.to_numeric(df["datadate"])

    fold1 = df[(df.datadate > 20100103) & (df.datadate <= 20181231)]
    fold2 = df[(df.datadate > 20190101) & (df.datadate <= 20191231)]
    fold3 = df[(df.datadate > 20200101) & (df.datadate <= 20201231)]
    ind1, ind2, ind3 = [], [], []
    for i in range(0, len(fold1) // 2):
        ind1.append(i)
        ind1.append(i)
    for i in range(0, len(fold2) // 2):
        ind2.append(i)
        ind2.append(i)
    for i in range(0, len(fold3) // 2):
        ind3.append(i)
        ind3.append(i)

    fold1.index = ind1
    fold2.index = ind2
    try:
        fold3.index = ind3[:len(fold3.index)]
    except ValueError:
        fold3.index = ind3[:len(fold3.index)]+[len(fold3) // 2 +2,len(fold3) // 2+2]
    return [[fold1, fold2, fold3]]