import pandas as pd
import numpy as np
import datetime

def get_historical_window_features(df, column, start_days_ago, end_days_ago):
    
    gb_cols = ['user_id', 'day_pod']
    _all_series = []
    
    for i in range(start_days_ago, end_days_ago, -1):

        _series = df.groupby(gb_cols)[column].shift(i)
        _all_series.append(_series)
    
    _df = pd.concat(_all_series, axis=1)
    col_labels = [f'{column}_T_minus_{i}' for i in range(start_days_ago, end_days_ago, -1)]
    _df.columns = col_labels
    
    return _df


def get_prediction_window_columns(df, column, prediction_window):
    
    gb_cols = ['user_id', 'day_pod']
    _all_series = []
    
    for i in range(prediction_window):

        _series = df.groupby(gb_cols)[column].shift(-i)
        _all_series.append(_series)
    
    _df = pd.concat(_all_series, axis=1)
    col_labels = [f'{column}_T{i}' for i in range(prediction_window)]
    _df.columns = col_labels
    
    return _df


def train_test_validation_split(df, label_column='total_consumption', recent_window=8*7, 
    lag_size=4*7, prediction_window=4*7, random_state=12345):
    
    # sort input DataFrame
    sort_cols = ['user_id', 'day_pod', 'start_time']
    df.sort_values(by=sort_cols, inplace=True)
    
    # base columns
    base_df = df[sort_cols]
    base_df.columns = ['user_id', 'day_pod', 'prediction_window_T0']
    
    # cyclical day of year features
    day_of_year_features_df = df[['day_of_year_sin', 'day_of_year_cos']]
    day_of_year_features_df.columns = ['day_of_year_sin_T0', 'day_of_year_cos_T0']
    
    # prediction window features 
    holidays_df = get_prediction_window_columns(df, 'holiday', prediction_window)
    min_temp_df = get_prediction_window_columns(df, 'min_temp', prediction_window)
    max_temp_df = get_prediction_window_columns(df, 'max_temp', prediction_window)
    
    # historical features
    recent_df = get_historical_window_features(df, label_column, 
        start_days_ago=recent_window+lag_size, end_days_ago=lag_size)
    prior_year_df = get_historical_window_features(df, label_column, 
        start_days_ago=365, end_days_ago=365-prediction_window)
    
    # labels
    labels_df = get_prediction_window_columns(df, label_column, prediction_window)
    
    final_dfs = [base_df, day_of_year_features_df, min_temp_df, max_temp_df, 
                 holidays_df, recent_df, prior_year_df, labels_df]
    final_df = pd.concat(final_dfs, axis=1)
    final_df['day_of_prediction'] =  final_df['prediction_window_T0'].dt.normalize() - datetime.timedelta(days=lag_size)
    
    print(f'Initial shape: {final_df.shape}')
    final_df.dropna(inplace=True)
    print(f'Shape after dropping na: {final_df.shape}')
    
    train, validate, test = np.split(final_df.sample(frac=1, random_state=random_state), 
                       [int(.6*len(final_df)), int(.8*len(final_df))])
    
    print(f'train shape: {train.shape}')
    print(f'validate shape: {validate.shape}')
    print(f'test shape: {test.shape}')
    
    assert len(set(train.index).intersection(set(validate.index))) == 0
    assert len(set(train.index).intersection(set(test.index))) == 0
    assert len(set(validate.index).intersection(set(test.index))) == 0
    
    return train, validate, test