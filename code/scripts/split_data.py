import pandas as pd

def simple_split_data(df, train_ratio=0.8):

    '''
        Goal: split data in trainings and test set with ratio

        Input:
        - df: df containing relevant assets,
        - train_ratio: ratio for split

        Output:
        - 2 df with correct split
    '''

    train_set_size = int(len(df) * train_ratio)
    df_training = df[:train_set_size]
    df_test = df[train_set_size:]
    return df_training, df_test

def manual_split(df, split_dates):

    '''
        Goal: split data in trainings and test set with specific split date

        Input:
        - df containing relevant assets, ratio for split
        - split_date: date for split

        Output:
        - 2 df with correct split
    '''


    if len(split_dates)==3:
        split_date_1, split_date_2, split_date_3 = split_dates
        df_training_1 = df.iloc[:split_date_1]
        df_val_1 = df.iloc[split_date_1:split_date_2]
        df_test_1 = df.iloc[split_date_2:split_date_3]
        df_training_2 = df.iloc[:split_date_2]
        df_val_2 = df_test_1
        df_test_2 = df.iloc[split_date_3:]
        return [[df_training_1, df_val_1, df_test_1], [df_training_2, df_val_2, df_test_2]]
    elif len(split_dates)==2:
        split_date_1, split_date_2 = split_dates
        df_training_1 = df.iloc[:split_date_1]
        df_val_1 = df.iloc[split_date_1:split_date_2]
        df_test_1 = df.iloc[split_date_2:]
        return [[df_training_1, df_val_1, df_test_1]]
    else:
        raise IndexError("wrong number of dates")




def simple_split_with_val(df,train_ratio=0.8, val_ratio=0.1):
    train_set_size = int(len(df) * train_ratio)
    val_set_size = int(len(df) * val_ratio)
    df_training = df[:train_set_size]
    df_validation = df[train_set_size:train_set_size+val_set_size]
    df_test = df[train_set_size+val_set_size:]
    return df_training, df_validation, df_test



