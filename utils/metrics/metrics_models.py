from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyperclip
from scipy import stats
import json

train_metrics_path = r'..\..\results\params_comparison\run_3\train\best\ordered_models\models.json'
test_metrics_path = r'..\..\results\params_comparison\run_3\test\ordered_models\models.json'
#path = train_metrics_path

def get_metrics(path):
    with open(path, 'r') as f:
        try:
            metrics_dict = json.load(f)
            try:
                metrics_dict = [values for values in metrics_dict.values()]
            except:
                pass
        except:
            metrics_dict = f.read().metrics_dict.replace("true", 'True').replace("false", 'False')
            metrics_dict = [values for values in metrics_dict.values()]
        return metrics_dict

def order_by_sum_of_normalization(unsorted):
    if len(unsorted) == 1:
        return unsorted
    mae, mape, rmse = [], [], []
    for metrics in unsorted:
        print('metrics', metrics)
        mae += [metrics['MAE']]
        mape += [metrics['MAPE']]
        rmse += [metrics['RMSE']]
    mae, mape, rmse = np.array(mae), np.array(mape), np.array(rmse)
    mae_norm = (mae - np.min(mae)) / (np.max(mae) - np.min(mae))
    mape_norm = (mape - np.min(mape)) / (np.max(mape) - np.min(mape))
    rmse_norm = (rmse - np.min(rmse)) / (np.max(rmse) - np.min(rmse))
    sum_norms = mae_norm + mape_norm + rmse_norm
    indexes = np.argsort(sum_norms)
    return indexes

metrics_train = get_metrics(train_metrics_path)
metrics_test = get_metrics(test_metrics_path)

# order dict by key
#metrics_train = sorted(metrics_train, key=lambda k: k['model_name'])
#metrics_test = sorted(metrics_test, key=lambda k: k['model_name'])

sort_order_train = order_by_sum_of_normalization(metrics_train)
metrics_train = [metrics_train[i] for i in sort_order_train]

sort_order_test = order_by_sum_of_normalization(metrics_test)
metrics_test = [metrics_test[i] for i in sort_order_test]

for metric in metrics_train:
    ae = np.abs(np.array(metric['real']) - np.array(metric['pred']))
    ape = np.abs(np.array(metric['real']) - np.array(metric['pred'])) / np.array(metric['real']) * 100
    se = (np.array(metric['real']) - np.array(metric['pred'])) ** 2
    metric['MAE_std'] = np.std(ae, ddof=1)
    metric['MAPE_std'] = np.std(ape, ddof=1)
    metric['RMSE_std'] = np.std(se, ddof=1) ** 0.5

for metric in metrics_test:
    ae = np.abs(np.array(metric['real']) - np.array(metric['pred']))
    ape = np.abs(np.array(metric['real']) - np.array(metric['pred'])) / np.array(metric['real']) * 100
    se = (np.array(metric['real']) - np.array(metric['pred'])) ** 2
    metric['MAE_std'] = np.std(ae, ddof=1)
    metric['MAPE_std'] = np.std(ape, ddof=1)
    metric['RMSE_std'] = np.std(se, ddof=1) ** 0.5

clipboard = ''
print('\n')
# for train in metrics_train:
#     mae_train = f'{train["MAE"]:.2f}±{train["MAE_std"]:.2f}'
#     mape_train = f'{train["MAPE"]:.2f}±{train["MAPE_std"]:.2f}'
#     rmse_train = f'{train["RMSE"]:.2f}±{train["RMSE_std"]:.2f}'
#     clipboard += f'{train["model_name"]}\t{mae_train}\t{mape_train}\t{rmse_train}\n'
for test in metrics_test:
    mae_test = f'{test["MAE"]:.2f}±{test["MAE_std"]:.2f}'
    mape_test = f'{test["MAPE"]:.2f}±{test["MAPE_std"]:.2f}'
    rmse_test = f'{test["RMSE"]:.2f}±{test["RMSE_std"]:.2f}'
    clipboard += f'{test["model_name"]}\t{mae_test}\t{mape_test}\t{rmse_test}\n'

print(clipboard)
pyperclip.copy(clipboard)