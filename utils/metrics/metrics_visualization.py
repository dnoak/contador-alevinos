import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d

metrics_paths = glob(r'results\detr_gside_gscale_conf\*')
#metrics_paths = list(filter(lambda x: 'yolov8n' in x, metrics_paths))

mae_list = []
mape_list = []
rmse_list = []
grid_scale_list = []
resize_scale_list = []
confiance_list = []
for metric_path in metrics_paths:
    with open(metric_path, 'r') as f:
        metrics = f.read().splitlines()
 
    mae_list.append(float([i for i in metrics if 'MAE' in i][0].split(': ')[1]))
    mape_list.append(float([i for i in metrics if 'MAPE' in i][0].split(': ')[1]))
    rmse_list.append(float([i for i in metrics if 'RMSE' in i][0].split(': ')[1]))

    grid_scale_list.append(float([i for i in metrics if 'grid_scale' in i][0].split(': ')[1]))
    resize_scale_list.append(float([i for i in metrics if 'resize_scale' in i][0].split(': ')[1]))
    confiance_list.append(float([i for i in metrics if 'confiance' in i][0].split(': ')[1]))

x = grid_scale_list
y = confiance_list

z1 = interp2d(x, y, mae_list)
z2 = interp2d(x, y, mape_list)
z3 = interp2d(x, y, rmse_list)

x_coords = np.linspace(min(x), max(x), len(set(x)))
y_coords = np.linspace(min(y), max(y), len(set(y)))

def plot_ax(idx, x, y, z, title):
    axs[idx].imshow(
        cv2.resize(z, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC),
        extent=[min(x), max(x), min(y), max(y)],
        origin='lower',
        cmap='hot')
    axs[idx].scatter(x, y, 400, facecolors='none')
    axs[idx].set_title(title)
    axs[idx].set_xticks(x)
    axs[idx].set_yticks(y)

fig, axs = plt.subplots(1, 3, figsize=(12, 6))
plot_ax(0, x, y, z1(x_coords, y_coords), 'MAE')
plot_ax(1, x, y, z2(x_coords, y_coords), 'MAPE')
plot_ax(2, x, y, z3(x_coords, y_coords), 'RMSE')
plt.show()