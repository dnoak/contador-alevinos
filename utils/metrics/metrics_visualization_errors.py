from glob import glob
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

metrics_paths = glob(r'..\..\results\best_params\all_models_test\*txt')
names_order = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
names_order+= ['rtdetr-l', 'rtdetr-x', 'detr-resnet-50', 'deformable-detr']
metric = 'mae'

save_path = Path(r'..\..\results\graphics\\') / f"{Path(metrics_paths[0]).parent.stem}_{metric}.png"

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
metrics = sorted(metrics, key=lambda x: names_order.index(x['model_name']))

def plot_ax(y, labels, fontsize):
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 5) 
    ax.boxplot([*y], widths=0.5, showfliers=False)
    normal = lambda x: np.random.normal(x, 0.04, len(y[0]))
    ax.scatter(list(map(normal, range(1, len(y)+1))), y, alpha=0.5, s=20) 
    ylabel = 'Absolute Percentage Error' if metric == 'mape' else 'Absolute Error'
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_yticklabels(range(0, 100, 1), fontsize=fontsize)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yscale('log', base=2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.16g}'))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

plot_ax(
    y=[m[metric]['values'] for m in metrics],
    labels=[m['model_name']+f'\n({metric.upper()}={np.mean(m[metric]["total"]):.2f})' for m in metrics],
    fontsize=12
)