import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

pwv = pd.read_csv("dummy_airwave_data.csv")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist(pwv[' brachial-femoral PWV (m/s)'], bins=100)
plt.title('Распределение скорости пульсовой волны')
plt.xlabel('Скорость пульсовой волны')
plt.ylabel('Количество пациентов')

corr = pwv.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sb.diverging_palette(220, 10, as_cmap=True)
sb.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()