import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import timeit
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.multi_modal import MultiModalData

# Игнорирование возникающих предупреждений.
warnings.filterwarnings('ignore')


def plot_series(df):
    df.plot(subplots=True, figsize=(15, 10))
    plt.xlabel("N")
    plt.legend(loc='best')
    plt.xticks(rotation='vertical')
    plt.show()


def comparsion_plot(pipeline, col_name, ts, old_predicted, new_predicted, train_len, start=0):
    pipeline.print_structure()
    plt.plot(range(start, len(ts)), ts[start:], label='Actual time series')
    plt.plot(range(train_len, len(ts)), old_predicted, label='Forecast before tuning')
    plt.plot(range(train_len, len(ts)), new_predicted, label='Forecast after tuning')
    plt.ylabel(col_name)
    plt.xlabel("N")
    plt.legend()
    plt.grid()
    plt.show()


def run_experiment_with_tuning(time_series, col_name, with_ar_pipeline=False, len_forecast=250, cv_folds=None):
    # with_ar_pipeline: is it needed to use pipeline with AR model or not
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(len_forecast))

    # divide on train and test
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    exog_arr = np.array(data['ΦN'])
    exog_train = exog_arr[:-len_forecast]
    exog_test = exog_arr[-len_forecast:]
    # Data for lagged transformation
    train_lagged = InputData(idx=np.arange(0, len(train_data)), features=train_data, target=train_data,
                             task=task, data_type=DataTypesEnum.ts)
    start_forecast = len(train_data)
    end_forecast = start_forecast + len_forecast
    predict_lagged = InputData(idx=np.arange(start_forecast, end_forecast), features=train_data, target=test_data,
                               task=task, data_type=DataTypesEnum.ts)

    # Data for exog operation
    train_exog = InputData(idx=np.arange(0, len(exog_train)), features=exog_train, target=train_data, task=task,
                           data_type=DataTypesEnum.ts)
    start_forecast = len(exog_train)
    end_forecast = start_forecast + len_forecast
    predict_exog = InputData(idx=np.arange(start_forecast, end_forecast), features=exog_test, target=test_data,
                             task=task, data_type=DataTypesEnum.ts)

    train_dataset = MultiModalData({'lagged': train_lagged, 'exog_ts_data_source': train_exog})

    predict_dataset = MultiModalData({'lagged': predict_lagged, 'exog_ts_data_source': predict_exog})

    # Get graph with several models and with arima pipeline
    if with_ar_pipeline:
        node_ar = PrimaryNode('ar')
        pipeline = Pipeline(node_ar)
    else:
        pipeline = get_complex_pipeline()

    old_predicted, new_predicted = make_forecast_with_tuning(pipeline, train_dataset, predict_dataset, task, cv_folds)

    old_predicted = np.ravel(np.array(old_predicted))
    new_predicted = np.ravel(np.array(new_predicted))
    test_data = np.ravel(test_data)

    mse_before = mean_squared_error(test_data, old_predicted, squared=False)
    mae_before = mean_absolute_error(test_data, old_predicted)
    print(f'RMSE before tuning - {mse_before:.4f}')
    print(f'MAE before tuning - {mae_before:.4f}\n')

    mse_after = mean_squared_error(test_data, new_predicted, squared=False)
    mae_after = mean_absolute_error(test_data, new_predicted)
    print(f'RMSE after tuning - {mse_after:.4f}')
    print(f'MAE after tuning - {mae_after:.4f}\n')

    comparsion_plot(pipeline, col_name, ts, old_predicted, new_predicted, len(train_data), start=0)

    start_point = len(time_series) - len_forecast * 2
    comparsion_plot(pipeline, col_name, ts, old_predicted, new_predicted, len(train_data), start_point)


def make_forecast_with_tuning(pipeline, train_input, predict_input, task, cv_folds):
    # Fit it
    start_time = timeit.default_timer()
    pipeline.fit(train_input)
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train pipeline\n')

    # Predict
    predicted_values = pipeline.predict(predict_input)
    old_predicted_values = predicted_values.predict

    pipeline_tuner = PipelineTuner(pipeline, task, iterations=15)
    pipeline = pipeline_tuner.tune_pipeline(input_data=train_input, loss_function=mean_squared_error,
                                            loss_params={'squared': False}, cv_folds=cv_folds, validation_blocks=3)
    # Fit pipeline on the entire train data and get prediction
    # pipeline.fit(train_input)
    # predicted_values = pipeline.predict(predict_input)
    new_predicted_values = predicted_values.predict

    return old_predicted_values, new_predicted_values


def get_complex_pipeline():
    node_lagged = PrimaryNode('lagged')
    node_exog = PrimaryNode('exog_ts_data_source')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged, node_exog])
    pipeline = Pipeline(node_ridge)
    return pipeline


data = pd.read_excel("kaggle/well_log.xlsx", sheet_name=0)
# убираем лишние столбцы
data.drop(columns=['SXO', 'Dtsyn', 'Vpsyn', 'sw new', 'sw new%', 'PHI2', 'ΔVp (m/s)'], inplace=True)
print(data)

plot_series(data)
plot_series(data[:100])

# рисуем корреляционную матрицу
corr = data.corr()
sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
plt.show()

# обходим массив и создаем недостающие значения глубины (остальные - NaN)
temp_df = pd.DataFrame()
count_columns = len(list(data.columns))
new_str = np.zeros(count_columns)
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
data = data.drop(columns=["index"])
print(data)

data = data.set_index('Depth')
data = data.interpolate(method='index')
data.reset_index(level=0, inplace=True)
print(data)

# после интерполяции удаляем "лишние" старые значения, чтобы остался ряд с шагом .15
inx_list = []
for i in range(1, len(data.Depth)):
    if data.Depth[i] not in depths_list:
        inx_list.append(i)

data = data.drop(index=inx_list)

plot_series(data[:100])

print(data)
print(data[:25])

data.to_csv('kaggle/well_log_interpolated.csv', index=False)

print(list(data.columns))
# Запуск.
for col in list(data.columns):
    ts = np.array(data[col])
    run_experiment_with_tuning(ts, col, with_ar_pipeline=False, len_forecast=500, cv_folds=2)
