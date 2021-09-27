import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_excel("kaggle/well_log.xlsx", sheet_name=0)
# убираем лишние столбцы
data.drop(columns=['SXO', 'Dtsyn', 'Vpsyn', 'sw new', 'sw new%', 'PHI2', 'ΔVp (m/s)'], inplace=True)
# print(data)

# рисуем корреляционную матрицу
# corr = data.corr()
# sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
# plt.show()

# обходим массив и создаем недостающие значения глубины (остальные - NaN)
counter = 0
temp_df = pd.DataFrame()

new_str = np.zeros(17)
new_str[:] = np.nan

for i in range(1, len(data.Depth)):
    if data.Depth[i] > data.Depth[i - 1] + 0.15:
        counter += 1
        new_str[0] = np.round(data.Depth[i-1] + 0.15, 2)  # добавляем новое значение для глубины
        str_to_pandas = pd.DataFrame([new_str], columns=list(data.columns))
        temp_df = temp_df.append(str_to_pandas)
        print(data.Depth[i], data.Depth[i - 1], str_to_pandas.Depth[0])

print(counter, "новых значений добавлено")
print(temp_df)

data = data.append(temp_df).sort_values(by=['Depth'])
print(data)

data.to_csv('kaggle/well_log_add_depths.csv', index=False)



