from glob import glob
from pathlib import Path
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

metrics_paths = glob(r'..\..\results\best_params\all_models_train/*.txt')
names_order = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
names_order += ['rtdetr-l', 'rtdetr-x', 'detr-resnet-50']
x_axis = 'real' 
y_axis = 'pred'

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
    }
    return metrics

metrics = list(map(get_txt_metrics, metrics_paths))
metrics = sorted(metrics, key=lambda x: names_order.index(x['model_name']))[:5]

def sub_plot_axs(ax, x, y, title, first):
    ax.scatter(x, y, alpha=0.5, s=20)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_regressed = intercept+slope*x
    ax.plot(x, y_regressed, 'r', c='red')
    #std = np.std(y-y_regressed)
    #ax.fill_between(x, y_regressed+std, y_regressed-std, alpha=0.2, color='red')
    
    ax.text(
        0.07, 0.95, 
        f"r² = {r_value**2:.3f}\nstd={std_err:.3f}", 
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#2060FF', alpha=0.15)
    )
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Real')

    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks(np.arange(0, max(x)+1, 60))
    ax.set_xticklabels(np.arange(0, max(x)+1, 60), fontsize=8)
    ax.set_yticks(np.arange(0, max(x)+1, 30))
    ax.set_yticklabels(np.arange(0, max(x)+1, 30), fontsize=8)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(x))

    if first:
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_ylabel('Predicted')

    ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax.set_title(title)


fig, ax = plt.subplots(1, len(metrics))
fig.set_size_inches(20, 4)
for i, m in enumerate(metrics):
    sub_plot_axs(
        ax[i], 
        np.array(m[x_axis]['values']), 
        np.array(m[y_axis]['values']),
        m['model_name'],
        first=i==0,
    )
#plt.autoscale(False)
plt.tight_layout()
plt.show()

