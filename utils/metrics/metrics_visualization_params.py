from pathlib import Path
import shutil
import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
import warnings
warnings.filterwarnings('ignore')

metrics_paths = glob(r'C:\Users\Luiz\Documents\TCC\contador-alevinos\results\metrics\(rtdetr-l-rtdetr-x)(g=0.3-0.5)(r=0.5-0.5)(c=0.4-0.7)(s=26)(seed=1011)\*')
model_name = 'rtdetr-x'
x_axis = 'grid_scale'
y_axis = 'confiance'
MAE_weight = 1
MAPE_weight = 1
RMSE_weight = 1

img_show = True
img_resize_scale = 1
save_path =  "../../results/best_params"

metrics_paths = list(filter(lambda x: model_name in x, metrics_paths))
print(f"{len(metrics_paths)} metrics loaded")

metrics_dict = {}
for metric_path in metrics_paths:
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

# sort x, y, z based on x
x, y, z_mae, z_mape, z_rmse = zip(*sorted(zip(x, y, z_mae, z_mape, z_rmse)))

x_coords = np.linspace(min(x), max(x), len(set(x)))
y_coords = np.linspace(min(y), max(y), len(set(y)))

z_mae = interp2d(x, y, get_list_from_dict(metrics_dict, 'mae'))(x_coords, y_coords)
z_mae_norm = (z_mae - np.min(z_mae)) / (np.ptp(z_mae))
z_mape = (interp2d(x, y,get_list_from_dict(metrics_dict, 'mape'))(x_coords, y_coords))
z_mape_norm = (z_mape - np.min(z_mape)) / (np.ptp(z_mape))
z_rmse = (interp2d(x, y,get_list_from_dict(metrics_dict, 'rmse'))(x_coords, y_coords))
z_rmse_norm = (z_rmse - np.min(z_rmse)) / (np.ptp(z_rmse))

best_all_metrics_coords = np.unravel_index(
    np.argmin(z_mae_norm*MAE_weight + z_mape_norm*MAPE_weight + z_rmse_norm*RMSE_weight),
    z_mae_norm.shape
)
print(f"\nbest_all_metrics_xy: {best_all_metrics_coords}")
best_all_metrics_x = sorted(list(set(x)))[best_all_metrics_coords[1]]
best_all_metrics_y = sorted(list(set(y)))[best_all_metrics_coords[0]]
print(f"best {x_axis}: {best_all_metrics_x}")
print(f"best {y_axis}: {best_all_metrics_y}")

best_all_metrics_path = [
    i for i in metrics_dict if 
    metrics_dict[i][x_axis] == best_all_metrics_x
    and
    metrics_dict[i][y_axis] == best_all_metrics_y
][0]

if save_path:
    shutil.copy(best_all_metrics_path, save_path)
    print(f"Saved at: {best_all_metrics_path}")

def plot_ax(idx, x, y, z, scale, title):
    H = z.shape[0]
    W = z.shape[1]
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
    plot_ax(0, x, y, z_mae_norm, img_resize_scale, 'MAE')
    plot_ax(1, x, y, z_mape_norm, img_resize_scale, 'MAPE')
    plot_ax(2, x, y, z_rmse_norm, img_resize_scale, 'RMSE')
    #set fig title
    fig.suptitle(f"{model_name} ({x_axis} x {y_axis})")
    plt.show()


