import sys
sys.path.append('../..')
from utils.common.image_utils import Image as im
import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

name_filter = 'rtdetr-l'
x_axis_metric = 'grid_scale'
y_axis_metric = 'confiance'

metrics_paths = glob(r'..\..\results\metrics\images\detr-ddetr-rtdetrl-rtdetrx-(samples=13)\*')
metrics_paths = list(filter(lambda x: name_filter in x, metrics_paths))

metrics_coords = []

for metric_path in metrics_paths:
    with open(metric_path, 'r') as f:
        metrics = f.read().splitlines()

    mae = float([i for i in metrics if 'MAE' in i][0].split(': ')[1])
    mape = float([i for i in metrics if 'MAPE' in i][0].split(': ')[1])
    rmse = float([i for i in metrics if 'RMSE' in i][0].split(': ')[1])
    x_value = float([i for i in metrics if x_axis_metric in i][0].split(': ')[1])
    y_value = float([i for i in metrics if y_axis_metric in i][0].split(': ')[1])

    metrics_dict = {
        'x': x_value,
        'y': y_value,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
    }
    metrics_coords.append(metrics_dict)
    
x_ticks = sorted(list(set([i['x'] for i in metrics_coords])))
y_ticks = sorted(list(set([i['y'] for i in metrics_coords])))

min_x_step = min(np.diff(x_ticks))
min_y_step = min(np.diff(y_ticks))

x_coords = np.linspace(min(x_ticks), max(x_ticks), int((max(x_ticks)-min(x_ticks))/min_x_step+1))
x_coords = np.round(x_coords, 2)
y_coords = np.linspace(min(y_ticks), max(y_ticks), int((max(y_ticks)-min(y_ticks))/min_y_step+1))
y_coords = np.round(y_coords, 2)

print(x_coords, y_coords)

image = np.zeros((len(set(x_coords)), len(set(y_coords))), dtype=np.float32)

for metrics in metrics_coords:
    coord_x = np.where(x_coords == metrics['x'])[0][0]
    coord_y = np.where(y_coords == metrics['y'])[0][0]
    image[coord_x, coord_y] = metrics['mae']
    print(metrics['mae'])

image = np.flipud(image)

fig, ax = plt.subplots()
ax.imshow(
    cv2.resize(image, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC),
    extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)],
    origin='lower',
    aspect='auto',
    cmap='hot')
ax.scatter([i['x'] for i in metrics_coords], [i['y'] for i in metrics_coords], 400, facecolors='none')
ax.set_title('MAE')
ax.set_xticks(x_coords)
ax.set_yticks(y_coords)
plt.show()

