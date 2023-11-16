from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

metrics_paths_train = glob(r'..\..\results\best_params\all_models_train/*.txt')
metrics_paths_test  = glob(r'..\..\results\best_params\all_models_test/*.txt')

x_axis = 'real' 
y_axis = 'pred'
xy_max = 330
fontsize = 13.5

save_path = Path(r'..\..\results\graphics\\') / "train_and_test_regressions.png"

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


def sub_plot_axs(ax, x, y, model_name, yaxis, first, last):
    ax.scatter(x, y, alpha=0.5, s=20)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_regressed = intercept+slope*x
    ax.plot(x, y_regressed, 'r', c='red')
    #ax.fill_between(x, y_regressed+std, y_regressed-std, alpha=0.2, color='red')
    ax.text(
        0.07, 0.95, 
        f"r² = {r_value**2:.3f}",
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
    
    ax.set_yticks(np.arange(0, xy_max, 80))
    ax.set_yticklabels(np.arange(0, xy_max, 80), fontsize=10)
    
    ax.set_xticks(np.arange(0, xy_max, 30))
    ax.set_xticklabels(np.arange(0, xy_max, 30), fontsize=10)
    
    ax.set_xlim(0, xy_max)
    ax.set_ylim(-5, xy_max)

    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    
    if model_name:
        ax.set_ylabel(model_name)
    
    if yaxis:
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_yticks(np.arange(0, xy_max, 80))
        ax.set_yticklabels(np.arange(0, xy_max, 80), fontsize=10)

    if first:
        ax.set_title(first, fontsize=fontsize)
        ax.xaxis.set_label_position('top')

    if last:
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xticks(np.arange(0, xy_max, 30))
        ax.set_xticklabels(np.arange(0, xy_max, 30), fontsize=10)

    ax.grid(True, which='both', linestyle='--', linewidth=0.4)


metrics_test = list(map(get_txt_metrics, metrics_paths_test))
metrics_train = list(map(get_txt_metrics, metrics_paths_train))
# order by r² from metrics_test
r2 = lambda x: stats.linregress(x['real']['values'], x['pred']['values'])[2]**2

metrics_test = sorted(metrics_test, key=r2, reverse=True)
sorted_metrics_train = []
for m in metrics_test:
    for i in metrics_train:
        if i['model_name'] == m['model_name']:
            sorted_metrics_train.append(i)
            break
metrics_train = sorted_metrics_train

fig, ax = plt.subplots(len(metrics_train), 2)
fig.set_size_inches(12, 16.3)

for i, m in enumerate(metrics_train):
    sub_plot_axs(
        ax[i, 0], 
        np.array(m[x_axis]['values']), 
        np.array(m[y_axis]['values']),
        model_name=m['model_name'],
        yaxis=True,
        first='Treino' if i==0 else False,
        last=i==len(metrics_train)-1
    )
for i, m in enumerate(metrics_test):
    sub_plot_axs(
        ax[i, 1], 
        np.array(m[x_axis]['values']), 
        np.array(m[y_axis]['values']),
        model_name=False,
        yaxis=False,
        first='Teste' if i==0 else False,
        last=i==len(metrics_test)-1
    )
#plt.autoscale(False)
#plt.subplots_adjust(wspace=0.07)
plt.tight_layout()
if save_path:
    plt.savefig(save_path, dpi=300)
plt.show()

