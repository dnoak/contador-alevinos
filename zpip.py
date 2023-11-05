import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('metrics.csv')

fig, ax = plt.subplots(figsize=(16, 9))
len_steps = len(df['validation/loss'])
loss = df['validation/loss']
# remove NaN for loss
loss = loss[~np.isnan(loss)]
ax.plot(range(len(loss)), loss, label='Validation loss')
ax.set_xlabel('Steps')
ax.set_ylabel('Loss')
ax.legend()
plt.show()

