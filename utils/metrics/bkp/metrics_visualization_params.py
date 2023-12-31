from pathlib import Path
import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
import warnings
warnings.filterwarnings('ignore')

metrics_paths = glob(r'F:\TCC\contagem-larvas\results\metrics\(yolov8n-yolov8s-yolov8m-yolov8l-yolov8x)(g=0.3-0.45)(r=0.5-0.5)(c=0.3-0.55)(s=130)(seed=1011)\*')
model_name = 'yolov8m'
x_axis = 'grid_scale'
y_axis = 'confiance'
MAE_weight = 1
MAPE_weight = 1
RMSE_weight = 1

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

z_mae_norm = (z_mae - np.min(z_mae)) / (np.max(z_mae) - np.min(z_mae))
z_mape_norm = (z_mape - np.min(z_mape)) / (np.max(z_mape) - np.min(z_mape))
z_rmse_norm = (z_rmse - np.min(z_rmse)) / (np.max(z_rmse) - np.min(z_rmse))
z_sum = z_mae_norm*MAE_weight + z_mape_norm*MAPE_weight + z_rmse_norm*RMSE_weight
pos_min_error = np.argmin(z_sum)
# search on dict the pos_min_error position
for i, (key, value) in enumerate(metrics_dict.items()):
    if i == pos_min_error:
        [print(f"{k}: {v}") for k, v in value.items()]
        break

z_mae =  interp2d(x, y, get_list_from_dict(metrics_dict, 'mae'))(x, y)
z_mape = interp2d(x, y, get_list_from_dict(metrics_dict, 'mape'))(x, y)
z_rmse = interp2d(x, y, get_list_from_dict(metrics_dict, 'rmse'))(x, y)

def plot_ax(idx, x, y, z, scale, title):
    z = (z - np.min(z)) / (np.max(z) - np.min(z))
    log_z = np.log(z+1)
    axs[idx].imshow(
        cv2.resize(log_z, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
        extent=[min(x), max(x), min(y), max(y)],
        origin='lower',
        aspect='auto',
        cmap='hot')
    if x_axis == 'grid_scale':
        x_label = 'Escala de Tiling'
    if y_axis == 'confiance':
        y_label = 'Confiança'
    axs[idx].scatter(x, y, 400, facecolors='none')
    axs[idx].set_title(title, fontsize=15)
    axs[idx].set_xticklabels(x, fontsize=15)
    axs[idx].set_xticks(x)
    axs[idx].set_yticklabels(y, fontsize=15)
    axs[idx].set_yticks(y)
    axs[idx].set_xlabel(x_label, fontsize=18)
    if idx == 0:
        axs[idx].set_ylabel(y_label, fontsize=18)
    
if img_show:
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    plot_ax(0, x, y, z_mae, img_resize_scale, 'MAE')
    plot_ax(1, x, y, z_mape, img_resize_scale, 'MAPE')
    plot_ax(2, x, y, z_rmse, img_resize_scale, 'RMSE')
    fig.suptitle(f"{model_name}", fontsize=20)
    plt.tight_layout()
    plt.show()


