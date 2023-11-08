from glob import glob
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

metrics_paths_train = glob(r'..\..\results\best_params\all_models_train\*txt')
metrics_paths_test = glob(r'..\..\results\best_params\all_models_test\*txt')

metric = 'mae'
fontsize = 13.5

save_path = Path(r'..\..\results\graphics\\') / f"train_and_test_{metric}.png"

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


def plot_ax(ax, y, x_label, y_label):
    ax.boxplot([*y], widths=0.5, showfliers=False)
    normal = lambda x: np.random.normal(x, 0.04, len(y[0]))
    ax.scatter(list(map(normal, range(1, len(y)+1))), y, alpha=0.5, s=20) 
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_yticklabels(range(0, 100, 1), fontsize=fontsize)
    ax.set_xticklabels(x_label, fontsize=fontsize)
    ax.set_yscale('log', base=2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.16g}'))

metrics_train = list(map(get_txt_metrics, metrics_paths_train))
metrics_train = sorted(metrics_train, key=lambda x: np.mean(x[metric]['total']))

metrics_test = list(map(get_txt_metrics, metrics_paths_test))
metrics_test = sorted(metrics_test, key=lambda x: np.mean(x[metric]['total']))

fig, ax = plt.subplots(2, 1)
plot_ax(
    ax[0],
    y=[m[metric]['values'] for m in metrics_train],
    x_label=[m['model_name']+f'\n({metric.upper()}={np.mean(m[metric]["total"]):.2f})' for m in metrics_train],
    y_label = 'Erro Percentual Absoluto (Treino)' if metric == 'mape' else 'Erro Absoluto (Treino)',
)
plot_ax(
    ax[1],
    y=[m[metric]['values'] for m in metrics_test],
    x_label=[m['model_name']+f'\n({metric.upper()}={np.mean(m[metric]["total"]):.2f})' for m in metrics_test],
    y_label = 'Erro Percentual Absoluto (Teste)' if metric == 'mape' else 'Erro Absoluto (Teste)',
)
fig.set_size_inches(18, 5) 
plt.tight_layout()
plt.show()