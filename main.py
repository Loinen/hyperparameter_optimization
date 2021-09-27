import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_excel("kaggle/well_log.xlsx", sheet_name=0)
# убираем лишние столбцы
data.drop(columns=['SXO', 'Dtsyn', 'Vpsyn', 'sw new', 'sw new%', 'PHI2', 'ΔVp (m/s)'], inplace=True)
# print(data)

data[:100].plot(subplots=True, figsize=(15, 10))
plt.xlabel("N")
plt.legend(loc='best')
plt.xticks(rotation='vertical')
plt.show()

# рисуем корреляционную матрицу
# corr = data.corr()
# sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
# plt.show()

# обходим массив и создаем недостающие значения глубины (остальные - NaN)
temp_df = pd.DataFrame()

new_str = np.zeros(17)
new_str[:] = np.nan

depth = data.Depth[0]
depths_list = []
while depth <= data.Depth.values[-1]:
    depth = np.round(depth + 0.15, 2)
    depths_list.append(depth)
    if depth not in data.Depth.values:
        new_str[0] = depth  # добавляем новое значение для глубины
        str_to_pandas = pd.DataFrame([new_str], columns=list(data.columns))
        temp_df = temp_df.append(str_to_pandas)

print(len(temp_df), "новых значений добавлено")

data = data.append(temp_df).sort_values(by=['Depth'])
data.reset_index(inplace=True)
print(data)

data = data.interpolate(method='index')

data[:100].plot(subplots=True, figsize=(15, 10))
plt.xlabel("N")
plt.legend(loc='best')
plt.xticks(rotation='vertical')
plt.show()

# после интерполяции удаляем "лишние" старые значения, чтобы остался ряд с шагом .15

inx_list = []
depth = data.Depth[0]
for i in range(1, len(data.Depth)):
    if data.Depth[i] not in depths_list:
        inx_list.append(i)

data = data.drop(index=inx_list)
data = data.drop(columns=["index"])

data[:100].plot(subplots=True, figsize=(15, 10))
plt.xlabel("N")
plt.legend(loc='best')
plt.xticks(rotation='vertical')
plt.show()

print(data)
print(data[:25])

data.to_csv('kaggle/well_log_interpolated.csv', index=False)



