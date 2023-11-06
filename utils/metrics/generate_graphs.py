import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

path_csvs = r'C:\Users\Luiz\Documents\TCC\contador-alevinos\data\models\result treino\*'

folders = glob.glob(path_csvs)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 10))
axes = axes.flatten()

for idx, folder in enumerate(folders):
    plot_title = ""
    full_path = ""
    epoch_column = ""
    loss_column = ""
    if ("yolov8" in folder or "rtdetr" in folder):
        full_path = os.path.join(folder, "results.csv")
        epoch_column = "                  epoch"
        loss_column = "           val/box_loss"
    else:
        full_path = os.path.join(folder, "metrics.csv")
        epoch_column = "epoch"
        loss_column = "validation_loss_bbox"

    data = pd.read_csv(full_path)
    axes[idx].plot(data[epoch_column], data[loss_column], marker='o', linestyle='-', color='blue')
    axes[idx].set_title(plot_title)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Train Box Loss')
    axes[idx].grid(True)

plt.tight_layout()
plt.show()