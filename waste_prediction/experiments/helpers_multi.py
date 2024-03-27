import json
import os
import time
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae
from darts.metrics import mape
from darts.metrics import rmse
from darts.metrics import r2_score
from waste_prediction.params import PROCESSED_DATA_PASS, RESULTS_PATH



def _compute_non_conformity_scores(fun, pred_series_list, actual_series):
    nc_scores = []
    conformal_prediction_horizon = len(pred_series_list[0])
    for i, pred_series in enumerate(pred_series_list):
        # print(pred_series.pd_dataframe())
        # print(actual_series[i*conformal_prediction_horizon:(i+1)*conformal_prediction_horizon].pd_dataframe())
        nc_scores.append(fun(pred_series, actual_series[i*conformal_prediction_horizon:(i+1)*conformal_prediction_horizon]))
    return nc_scores


def _conformal_quantile(nc_scores, alpha):
    n_cal = len(nc_scores)
    return np.quantile(nc_scores, np.ceil((1-alpha)*(n_cal+1))/n_cal)


def _empirical_coverage(nc_scores, quantile):
    print(nc_scores)
    n_cov = len(nc_scores)  
    return np.sum(nc_scores < quantile)/n_cov


def _evaluate_model(train_series, test_series, model_list, output_dir, conformal_alpha, whole_series, calibration_series, coverage_series):
    # Normalize the dataset
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)
    whole_scaled = scaler.transform(whole_series)

    train_scaled_df = train_scaled.pd_dataframe().reset_index()[['ticket_date', 'net_weight_kg']]
    test_df = test_series.pd_dataframe().reset_index()[['ticket_date', 'net_weight_kg']]
    whole_sclaed_df = whole_scaled.pd_dataframe().reset_index()[['ticket_date', 'net_weight_kg']]
    whole_df = whole_series.pd_dataframe().reset_index()[['ticket_date', 'net_weight_kg']]
    calibration_df = calibration_series.pd_dataframe().reset_index()[['ticket_date', 'net_weight_kg']]
    coverage_df = coverage_series.pd_dataframe().reset_index()[['ticket_date', 'net_weight_kg']]

    # Find the day of the week
    train_scaled_df['day_of_week'] = train_scaled_df['ticket_date'].dt.dayofweek
    test_df['day_of_week'] = test_df['ticket_date'].dt.dayofweek
    whole_sclaed_df['day_of_week'] = whole_df['ticket_date'].dt.dayofweek
    calibration_df['day_of_week'] = calibration_df['ticket_date'].dt.dayofweek

    # Divide days into buckets
    train_day_series_list = []
    for i in range(7):
        temp_df = train_scaled_df.loc[train_scaled_df['day_of_week'] == i].copy()[['ticket_date', 'net_weight_kg']].reset_index(drop=True)
        train_day_series_list.append(temp_df)
        
    whole_day_series_list = []
    for i in range(7):
        temp_df = whole_sclaed_df.loc[whole_sclaed_df['day_of_week'] == i].copy()[['ticket_date', 'net_weight_kg']].reset_index(drop=True)
        whole_day_series_list.append(temp_df)

    # Fit
    print('Fitting models...')
    training_start_time = time.time()
    for i in tqdm(range(7), desc='Fit'):
        model_list[i].fit(series=TimeSeries.from_dataframe(train_day_series_list[i], time_col='ticket_date', value_cols='net_weight_kg'))
    training_end_time = time.time()
    print('Models fitted!')
    
    # Predict test
    print('Predicting test...')
    test_pred_rows = []

    test_day_number_list = test_df['day_of_week'].tolist()
    test_ticket_date_list = test_df['ticket_date'].tolist()

    model_prediction_list = []
    predicting_start_time = time.time()
    for i in tqdm(range(7), desc='Predict target'):
        predictions = model_list[i].predict(n=len(test_series))
        model_prediction_list.append(scaler.inverse_transform(predictions))
    predicting_end_time = time.time()

    for i in range(len(test_series)):
        week_index = int(i / 7)
        test_pred_value = model_prediction_list[test_day_number_list[i]][week_index].pd_dataframe().reset_index()['net_weight_kg'].tolist()[0]
        test_pred_rows.append([test_ticket_date_list[i], test_pred_value])

    test_pred_series = pd.DataFrame(test_pred_rows, columns=['ticket_date', 'net_weight_kg'])
    test_pred_series = TimeSeries.from_dataframe(test_pred_series, time_col='ticket_date', value_cols='net_weight_kg')
    print('Test predicted!')
    
    # Conformal calibration
    raise NotImplementedError("Conformal calibration was not implemented for multi-model prediction.")
    calibration_pred_rows = []
    
    calibration_day_number_list = calibration_df['day_of_week'].tolist()
    calibration_ticket_date_list = calibration_df['ticket_date'].tolist()
    
    print(f'Predicting conformal calibration over horizon of {1} days...')
    model_prediction_list_calibration = [[] for _ in range(7)]
        
    start_past_time = whole_series.start_time()
    end_past_time = calibration_series.start_time() - pd.Timedelta(days=1)
    while end_past_time <= calibration_series.end_time()-pd.Timedelta(days=7):
        for i in range(7):
            day_number = start_past_time.dayofweek + i
            pred = model_list[day_number].predict(n=1, series=whole_day_series_list[day_number].loc(start_past_time <= whole_day_series_list[day_number] and whole_day_series_list[day_number] <= end_past_time))
            model_prediction_list_calibration[day_number].append(scaler.inverse_transform(pred))
        end_past_time = end_past_time + pd.Timedelta(days=7)
        start_past_time = start_past_time + pd.Timedelta(days=7)
        
    # Reassemble the predictions into one series
    for i in range(len(calibration_series)):
        week_index = int(i / 7)
        calibration_pred_value = model_prediction_list_calibration[calibration_day_number_list[i]][week_index].pd_dataframe().reset_index()['net_weight_kg'].tolist()[0]
        print("calibration_pred_value", calibration_pred_value)
        raise
        calibration_pred_rows.append([calibration_ticket_date_list[i], calibration_pred_value])

    # Rename variables
    # actual_series = TimeSeries.from_dataframe(test_series, time_col='ticket_date', value_cols='net_weight_kg')
    actual_series = test_series
    pred_series = test_pred_series

    # ---- Save predictions

    actual_series_df = actual_series.pd_dataframe().reset_index()[['ticket_date', 'net_weight_kg']]
    actual_series_df = actual_series_df[['ticket_date', 'net_weight_kg']].rename(columns={'net_weight_kg': 'actual_net_weight_kg'})
    pred_series_df = pred_series.pd_dataframe().reset_index()
    pred_series_df = pred_series_df[['ticket_date', 'net_weight_kg']].rename(columns={'net_weight_kg': 'predicted_net_weight_kg'})

    output_df = pd.merge(actual_series_df, pred_series_df, on='ticket_date')

    def get_history_over_time(fun):
        hist = []
        for i in range(len(actual_series)):
            hist_val = fun(actual_series[i: i + 1], pred_series[i: i + 1])
            hist.append(hist_val)
        return hist

    output_df['rmse'] = get_history_over_time(rmse)
    output_df['mae'] = get_history_over_time(mae)

    # ---- Save summary

    mape_val = None
    try:
        mape_val = mape(actual_series, pred_series)
    except:
        pass

    summary = {
        'rmse': rmse(actual_series, pred_series),
        'mae': mae(actual_series, pred_series),
        'mape': mape_val,
        'r2_score': r2_score(actual_series, pred_series),
        'training_time': training_end_time - training_start_time,
        'predicting_time': predicting_end_time - predicting_start_time
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f)

    output_df.to_csv(os.path.join(output_dir, 'output.csv'), index=False)

    print(summary)


def run(params, generate_model_name, generate_model, conformal_alpha=0.05, conformal_prediction_horizon=1):
    dataset_name = params['dataset_name']
    test_calibration_split_before = params['test_calibration_split_before']
    empirical_coverage_split_before = params['empirical_coverage_split_before']
    only_weekdays = params['only_weekdays']
    is_differenced = params['is_differenced']

    # Config
    daily_waste_data_file_path = os.path.join(PROCESSED_DATA_PASS, dataset_name, 'imputed_data.csv')
    result_output_dir_path = os.path.join(RESULTS_PATH, dataset_name)

    # Create result dir
    if not os.path.exists(result_output_dir_path):
        os.makedirs(result_output_dir_path)

    # Preprocess
    df = pd.read_csv(daily_waste_data_file_path)
    df['ticket_date'] = df['ticket_date'].astype('datetime64[ns]')

    # Merge weekends to Monday
    if only_weekdays:
        df['year'] = df['ticket_date'].dt.year
        df['week'] = df['ticket_date'].dt.strftime('%U').astype(int)
        df['day_of_week'] = df['ticket_date'].dt.dayofweek

        # Tue, Wed, Thu, Fri
        df_part_1 = df.loc[df['day_of_week'].isin([1, 2, 3, 4])].copy().reset_index(drop=True)

        # Mon
        df_part_2 = df.loc[df['day_of_week'] == 0].copy().reset_index(drop=True)

        # Sat, Sun
        df_part_3 = df.loc[df['day_of_week'].isin([5, 6])].copy().reset_index(drop=True)
        df_part_3_copy = df_part_3.copy()
        df_part_3['week'] = df_part_3['week'] + 1
        df_part_3 = df_part_3.groupby(['year', 'week']).agg('sum').reset_index()
        df_part_3 = df_part_3[['year', 'week', 'net_weight_kg']].rename({'net_weight_kg': 'weekend_net_weight_kg'}, axis='columns')

        df_part_23 = pd.merge(df_part_2, df_part_3, how='left', left_on=['year', 'week'], right_on=['year', 'week'])

        # Add weekend weight to Monday
        df_part_23['net_weight_kg'] = df_part_23['net_weight_kg'] + df_part_23['weekend_net_weight_kg']

        # Merge all data back together
        del df_part_3_copy['net_weight_kg']
        df = pd.concat([df_part_1, df_part_23, df_part_3_copy])

        df = df[['ticket_date', 'net_weight_kg']].sort_values(by=['ticket_date'], ascending=True).reset_index(drop=True)

        # Fill missing values - linear
        df[['net_weight_kg']] = df[['net_weight_kg']].fillna(value=0)

    # Apply differencing
    if is_differenced:
        df['net_weight_kg_diff'] = df['net_weight_kg'].diff(periods=1).fillna(df['net_weight_kg'])
        df['net_weight_kg'] = df['net_weight_kg_diff']
        del df['net_weight_kg_diff']

    # Models
    model_list = []
    for i in range(7):
        model = generate_model(params)
        model_list.append(model)

    # Model name
    model_name = generate_model_name(params)

    # Output directory path
    output_dir = f'{result_output_dir_path}/{model_name}'

    # Check if output dir and params exists
    if os.path.exists(output_dir) and os.path.exists(f'{output_dir}/summary.json'):
        print('Skip: {} - {}'.format(dataset_name, model_name))
        return
    else:
        print('Run: {} - {}'.format(dataset_name, model_name))

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert to a Darts Timeseries
    series = TimeSeries.from_dataframe(df, time_col='ticket_date', value_cols='net_weight_kg')

    # Split into train and test series
    train_series, test_series = series.split_before(pd.Timestamp(test_calibration_split_before))
    calibration_series, empirical_coverage_series = test_series.split_before(pd.Timestamp(empirical_coverage_split_before))

    # Evaluate test
    _evaluate_model(train_series, test_series, model_list, output_dir, conformal_alpha=conformal_alpha, whole_series=series, calibration_series=calibration_series, coverage_series=empirical_coverage_series)

    # Save params
    with open(f'{output_dir}/params.json', 'w') as f:
        json.dump(params, f)

    # Clear figures
    plt.close('all')
