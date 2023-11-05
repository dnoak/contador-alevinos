from glob import glob
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

metrics_paths = glob(r'..\..\results\best_params\all_models\*')
names_order = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
names_order+= ['rtdetr-l', 'rtdetr-x']
metric = 'mae'
 
def get_txt_metrics(path):
    with open(path, 'r') as f:
        metrics = f.read().splitlines()
    model_name = Path(path).stem.split('_')[0]
    metrics = {
        'mae': {
            'total': float([i for i in metrics if 'MAE'  in i][0].split(maxsplit=2)[1]),
            'values': eval([i for i in metrics if 'MAE'  in i][0].split(maxsplit=2)[2]),
        },
        'mape': {
            'total': float([i for i in metrics if 'MAPE' in i][0].split(maxsplit=2)[1]),
            'values': eval([i for i in metrics if 'MAPE' in i][0].split(maxsplit=2)[2]),
        }
    }
    return model_name, metrics

metrics = list(map(get_txt_metrics, metrics_paths))
metrics = sorted(metrics, key=lambda x: names_order.index(x[0]))
metrics = [[m[0]+f"\n({metric.upper()}={m[1][metric]['total']:.2f})", m[1]] for m in metrics]
[print(i) for i in metrics]

def plot_ax(y, labels):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4) 
    ax.boxplot([*y], widths=0.5, showfliers=False)
    normal = lambda x: np.random.normal(x, 0.04, len(y[0]))
    ax.scatter(list(map(normal, range(1, len(y)+1))), y, alpha=0.5, s=20) 
    ax.set_xticklabels(labels)
    ylabel = 'Absolute Percentage Error' if metric == 'mape' else 'Absolute Error'
    ax.set_ylabel(ylabel)        
    ax.set_xlabel('Model')
    ax.set_yscale('log', base=2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.16g}'))
    plt.show()

plot_ax(
    y=[i[1][metric]['values'] for i in metrics],
    labels=[i[0] for i in metrics],
)