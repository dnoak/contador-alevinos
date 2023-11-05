from glob import glob
from pathlib import Path
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

metrics_paths = glob(r'..\..\results\best_params\(yolov8n-yolov8s-yolov8m-yolov8l-yolov8x)(g=0.3-0.45)(r=0.5-0.5)(c=0.3-0.55)(s=130)(seed=1011)/*.txt')
random.shuffle(metrics_paths)
metrics_paths = metrics_paths[:5]
names_order = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
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

def sub_plot_axs(ax, x, y, mae, title):
    ax.scatter(x, y, alpha=0.5, s=20)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_regressed = intercept+slope*x
    ax.plot(x, y_regressed, 'r', c='red')
    #std = np.std(y-y_regressed)
    #ax.fill_between(x, y_regressed+std, y_regressed-std, alpha=0.2, color='red')
    ax.set_xticks(np.arange(0, max(x)+1, max(x)//5))
    ax.set_yticks(np.arange(0, max(y)+1, max(y)//6))

    ax.set_xlabel('Real')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    
fig, ax = plt.subplots(1, 5)
fig.set_size_inches(20, 4)
for i, m in enumerate(metrics):
    sub_plot_axs(
        ax[i], 
        np.array(m[x_axis]['values']), 
        np.array(m[y_axis]['values']),
        m['mae']['total'], 
        m['model_name']
    )
#plt.tight_layout()
plt.show()

