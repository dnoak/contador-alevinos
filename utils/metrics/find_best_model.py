from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

metrics_paths = glob(r'..\..\results\best_params\all_models_train/*.txt')
names_order = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
names_order += ['rtdetr-l', 'rtdetr-x', 'detr-resnet-50', 'deformable-detr']
x_axis = 'real' 
y_axis = 'pred'
xy_max = 330

def search_in_lines(lines, key, maxsplit, position):
    return {
        'total': float([i for i in lines if key in i][0].split(maxsplit=maxsplit)[position]),
        'values': eval([i for i in lines if key in i][0].split(maxsplit=maxsplit)[position+1]),
    }
def get_txt_metrics(path):
    with open(path, 'r') as f:
        metrics = f.read().splitlines()
    metrics = {
        'model_name': Path(path).stem.split('_')[0],
        'pred': search_in_lines(metrics, 'pred', 2, 1),
        'real': search_in_lines(metrics, 'real', 2, 1),
        'mae':  search_in_lines(metrics, 'MAE',  2, 1),
        'mape': search_in_lines(metrics, 'MAPE', 2, 1),
        'rmse': search_in_lines(metrics, 'RMSE', 2, 1),
    }
    return metrics

metrics = list(map(get_txt_metrics, metrics_paths))
metrics_original = metrics.copy()
#metrics = sorted(metrics, key=lambda x: names_order.index(x['model_name']))

def norm_mae_mape_rmse(metrics):
    maes = np.array([i['mae']['total'] for i in metrics])
    mapes = np.array([i['mape']['total'] for i in metrics])
    rmses = np.array([i['rmse']['total'] for i in metrics])
    maes_norm = (maes - np.min(maes)) / (np.max(maes) - np.min(maes))
    mapes_norm = (mapes - np.min(mapes)) / (np.max(mapes) - np.min(mapes))
    rmses_norm = (rmses - np.min(rmses)) / (np.max(rmses) - np.min(rmses))
    for i, metric in enumerate(metrics):
        metric['mae']['norm'] = maes_norm[i]
        metric['mape']['norm'] = mapes_norm[i]
        metric['rmse']['norm'] = rmses_norm[i] 
    return metrics

metrics_norm = norm_mae_mape_rmse(metrics)
metrics_norm = sorted(metrics_norm, key=lambda x: x['mae']['norm'] + x['mape']['norm'] + x['rmse']['norm'])
metrics = sorted(metrics, key=lambda x: metrics_norm.index(x)) 
                 
rows = []
for metric in metrics_norm:
    model =f"{metric['model_name']}\t"
    mae = f"{metric['mae']['total']:.2f}\t"
    mape = f"{metric['mape']['total']:.2f}\t"
    rmse = f"{metric['rmse']['total']:.2f}"
    rows.append([model, mae, mape, rmse])

clipboard = '\n'.join([''.join(row) for row in rows])
import pyperclip
pyperclip.copy(clipboard)
print(clipboard)
