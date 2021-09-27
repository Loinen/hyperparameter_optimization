import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_excel("kaggle/well_log.xlsx", sheet_name=0)
data.drop(columns=['SXO', 'Dtsyn', 'Vpsyn', 'sw new', 'sw new%', 'PHI2', 'ΔVp (m/s)'], inplace=True)
print(data)

corr = data.corr()
sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
plt.show()




