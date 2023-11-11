from glob import glob
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

metrics_paths_train = glob(r'..\..\results\best_params\all_models_train\*txt')
metrics_paths_test = glob(r'..\..\results\best_params\all_models_test\*txt')

metric = 'mape'
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
    ax.set_yscale('symlog', base=2)
    ax.set_ylim(-1/4)
    ax.grid(True, which='both', linestyle='--', linewidth=0.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.16g}'))


metrics_test = list(map(get_txt_metrics, metrics_paths_test))
metrics_train = list(map(get_txt_metrics, metrics_paths_train))

# metrics_test = sorted(metrics_test, key=lambda x: np.mean(x[metric]['total']))
metrics_test = sorted(metrics_test, key=lambda x: np.mean(x[metric]['total']))

sorted_metrics_train = []
for m in metrics_test:
    for i in metrics_train:
        if m['model_name'] == i['model_name']:
            sorted_metrics_train.append(i)
            break
metrics_train = sorted_metrics_train

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
fig.set_size_inches(18, 10) 
plt.tight_layout()
if save_path:
    plt.savefig(save_path, dpi=300)
plt.show()


import numpy as np
m1=[342,366,360,359,365,397,397,705,747]
m2=[416,438,453,506,612,628,731,710,831]
p1=[5.52,5.45,5.29,5.14,5.22,5.75,5.75,9.74,11.02]
p2=[6.5,16.38,6.58,7.97,9.45,7.93,10.59,9.70,9.98]

r1=[4.65  ,5.16  ,5.49  ,5.80  ,5.60  ,6.38  ,6.54  ,9.35  ,12.35] 
r2=[5.77,  6.74,  6.53,  7.17,  8.21,  10.68, 10.16, 11.89, 13.06] 

k=np.abs(np.array(r1)-np.array(r2))
print(np.round(np.mean(k),2))
