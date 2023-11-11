from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FuncFormatter

#train_labels_path = glob(r'..\..\data\datasets\train_val\yolov8_640x640_train=2689_val=676\train\labels\*.txt')
#val_labels_path   = glob(r'..\..\data\datasets\train_val\yolov8_640x640_train=2689_val=676\valid\labels\*.txt')
train_labels_path = glob(r'..\..\data\datasets\train_val\yolov8_originalres_train=130_val=0\train\labels\*.txt')
val_labels_path   = glob(r'..\..\data\datasets\test\yolov8_originalres_test=32\test\labels\*.txt')
nbins = 16

def len_annotations(labels_paths):
    counts = []
    for p in labels_paths:
        with open(p, 'r') as f:
            annotations = f.read().splitlines()
        #if len(annotations) > 0:
        counts.append(len(annotations))
    return counts

counts_train = len_annotations(train_labels_path)
counts_val   = len_annotations(val_labels_path)

fig, ax = plt.subplots()
fig.set_size_inches(7, 5)
ax.hist(counts_train, bins=nbins, alpha=0.5, label='treino')
ax.hist(counts_val, bins=nbins, alpha=0.5, label='teste')
ax.legend(loc='upper right')
ax.set_xlabel('Número de peixes', fontsize=13.5)
ax.set_ylabel('Frequência', fontsize=13.5)
ax.set_title('Histograma do número de anotações por imagem', fontsize=13.5)
ax.grid(True, which='both', linestyle='--', linewidth=0.2)
#ax.set_yscale('log', base=2)
#ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.16g}'))
plt.tight_layout()
plt.show()

