from pathlib import Path
import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
import warnings
warnings.filterwarnings('ignore')

metrics_paths = glob(r'C:\Users\Luiz\Documents\TCC\contador-alevinos\results\metrics\(detr-resnet-50)(g=0.2-0.5)(r=0.5-0.5)(c=0.3-0.9)(s=32)(seed=1011)\*')
model_name = 'detr-resnet-50'
x_axis = 'grid_scale'
y_axis = 'confiance'
MAE_weight = 1
MAPE_weight = 1
RMSE_weight = 1

img_show = True
img_show = True
img_resize_scale = 1

print(f"{len(metrics_paths)} metrics loaded")

metrics_dict = {}
for metric_path in metrics_paths:
    if model_name not in Path(metric_path).stem:
        continue
    with open(metric_path, 'r') as f:
        metrics = f.read().splitlines()
    metrics_dict[metric_path] = {
        'mae':  float([i for i in metrics if 'MAE'  in i][0].split()[1]),
        'mape': float([i for i in metrics if 'MAPE' in i][0].split()[1]),
        'rmse': float([i for i in metrics if 'RMSE' in i][0].split()[1]),
        'grid_scale': float([i for i in metrics if 'grid_scale' in i][0].split()[1]),
        'resize_scale': float([i for i in metrics if 'resize_scale' in i][0].split()[1]),
        'confiance': float([i for i in metrics if 'confiance' in i][0].split()[1]),
    }

get_list_from_dict = lambda dict, key: [dict[i][key] for i in dict]

x = get_list_from_dict(metrics_dict, x_axis)
y = get_list_from_dict(metrics_dict, y_axis)
z_mae = get_list_from_dict(metrics_dict, 'mae')
z_mape = get_list_from_dict(metrics_dict, 'mape')
z_rmse = get_list_from_dict(metrics_dict, 'rmse')

z_mae =  interp2d(x, y, get_list_from_dict(metrics_dict, 'mae'))(x, y)
z_mape = interp2d(x, y, get_list_from_dict(metrics_dict, 'mape'))(x, y)
z_rmse = interp2d(x, y, get_list_from_dict(metrics_dict, 'rmse'))(x, y)

def plot_ax(idx, x, y, z, scale, title):
    z = (z - np.min(z)) / (np.max(z) - np.min(z))
    log_z = np.log(z+1)
    axs[idx].imshow(
        cv2.resize(log_z, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4),
        extent=[min(x), max(x), min(y), max(y)],
        origin='lower',
        aspect='auto',
        cmap='hot')
    axs[idx].scatter(x, y, 400, facecolors='none')
    axs[idx].set_title(title)
    axs[idx].set_xticks(x)
    axs[idx].set_yticks(y)
    axs[idx].set_xlabel(x_axis)
    if idx == 0:
        axs[idx].set_ylabel(y_axis)
    
if img_show:
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    plot_ax(0, x, y, z_mae, img_resize_scale, 'MAE')
    plot_ax(1, x, y, z_mape, img_resize_scale, 'MAPE')
    plot_ax(2, x, y, z_rmse, img_resize_scale, 'RMSE')
    fig.suptitle(f"{model_name} ({x_axis} x {y_axis})")
    plt.show()


