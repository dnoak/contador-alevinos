from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyperclip
from scipy import stats
import json

train_metrics_path = r'..\..\results\params_comparison\run_4\test\ordered_models\models.json'
test_metrics_path = r'..\..\results\params_comparison\run_4\train\ordered_models\models.json'
path = train_metrics_path

def get_metrics(path):
    with open(path, 'r') as f:
        return json.load(f)

def order_by_sum_of_normalization(unsorted):
    if len(unsorted) == 1:
        return unsorted
    mae, mape, rmse = [], [], []
    for metrics in unsorted:
        mae += [metrics['MAE']]
        mape += [metrics['MAPE']]
        rmse += [metrics['RMSE']]
    mae, mape, rmse = np.array(mae), np.array(mape), np.array(rmse)
    mae_norm = (mae - np.min(mae)) / (np.max(mae) - np.min(mae))
    mape_norm = (mape - np.min(mape)) / (np.max(mape) - np.min(mape))
    rmse_norm = (rmse - np.min(rmse)) / (np.max(rmse) - np.min(rmse))
    sum_norms = mae_norm + mape_norm + rmse_norm
    indexes = np.argsort(sum_norms)
    return [unsorted[i] for i in indexes]

metrics = get_metrics(path)
for metric in metrics:
    print(metric['model_name'])
    print(metric['pred'])
    print(metric['real'])
    ae = np.abs(np.array(metric['real']) - np.array(metric['pred']))
    ape = np.abs(np.array(metric['real']) - np.array(metric['pred'])) / np.array(metric['real']) * 100
    se = (np.array(metric['real']) - np.array(metric['pred'])) ** 2
    metric['MAE_std'] = np.std(ae)
    metric['MAPE_std'] = np.std(ape)
    metric['RMSE_std'] = np.std(se) ** 0.5

metrics = order_by_sum_of_normalization(metrics)

clipboard = ''
print('\n', path)
for metric in metrics:
    mae = f'{metric["MAE"]:.2f} ±{metric["MAE_std"]:.2f}'
    mape = f'{metric["MAPE"]:.2f} ±{metric["MAPE_std"]:.2f}'
    rmse = f'{metric["RMSE"]:.2f} ±{metric["RMSE_std"]:.2f}'
    clipboard += f'{mae}\t{mape}\t{rmse}\n'

print(clipboard)
pyperclip.copy(clipboard)