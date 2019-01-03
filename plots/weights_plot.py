import pandas as pd

train = pd.read_csv('../logs/mtn_model_0/train_log.csv')

targets = {'genre_weight'            : 'Genre',
           'subgenres_weight'        : 'Subgenres',
           'mfcc_weight'             : 'MFCC',
           'chroma_weight'           : 'Chroma',
           'spectral_contrast_weight': 'Spectral contrast'}

df = train[['epoch', 'iteration', *targets.keys()]]
max_iter = df['iteration'].max()
max_epoch = df['epoch'].max()

df['timestep'] = df['epoch'] * max_iter + df['iteration']

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

fig, ax = plt.subplots(figsize=(14, 12))

for k in targets.keys():
    sns.lineplot(x='timestep', y=k, data=df, ax=ax, linewidth=5)

sns.despine(fig, ax, trim=True)
plt.legend(labels=[*targets.values()], fontsize=26)

plt.xlabel('Epoch', fontsize=26)

indices = [0] + [*range(4, max_epoch + 1, 5)]
labels = [i + 1 for i in indices]

plt.xticks([i * max_iter for i in indices], labels, fontsize=24)

plt.ylabel('Weight', fontsize=26)
plt.yticks(fontsize=24)

plt.show()
