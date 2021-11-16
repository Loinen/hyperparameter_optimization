import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import timeit
import seaborn as sns
from datetime import datetime

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


def comparsion_plot(col_name, ts, old_predicted, new_predicted, train_len, start=0, title="Comparsion plot"):
    now_dt = datetime.now()
    now = now_dt.strftime("%d_%m_%Y_%H-%M-%S")

    plt.plot(range(start, len(ts)), ts[start:], label='Actual time series')
    plt.plot(range(train_len, len(ts)), old_predicted, label='Forecast before tuning', linestyle='--', color='c')
    plt.plot(range(train_len, len(ts)), new_predicted, label='Forecast after tuning', color='g')
    plt.title(title)
    plt.ylabel(col_name)
    plt.xlabel("N")
    plt.legend()
    plt.grid()
    plt.savefig('results/{title}_{now}'.format(title=title, now=now))
    plt.show()


def count_errors(test_data, old_predicted, new_predicted):
    mse_before = mean_squared_error(test_data, old_predicted, squared=False)
    mae_before = mean_absolute_error(test_data, old_predicted)
    print(f'RMSE before tuning - {mse_before:.4f}')
    print(f'MAE before tuning - {mae_before:.4f}\n')

    mse_after = mean_squared_error(test_data, new_predicted, squared=False)
    mae_after = mean_absolute_error(test_data, new_predicted)
    print(f'RMSE after tuning - {mse_after:.4f}')
    print(f'MAE after tuning - {mae_after:.4f}\n')

    return round(mse_before, 2), round(mae_before), round(mse_after), round(mae_after)


def prepare_input_data(features_train_data, target_train_data, features_test_data, target_test, len_forecast, task):
    train_input = InputData(idx=np.arange(0, len(features_train_data)), features=features_train_data,
                            target=target_train_data, task=task, data_type=DataTypesEnum.ts)
    start_forecast = len(features_train_data)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast), features=features_test_data,
                              target=target_test, task=task, data_type=DataTypesEnum.ts)

    return train_input, predict_input


def run_experiment_with_tuning(time_series, col_name, len_forecast=250, cv_folds=None):
    # with_ar_pipeline: is it needed to use pipeline with AR model or not
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(len_forecast))

    # divide on train and test
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    exog_arr = np.array(data['ΦN'])
    exog_train = exog_arr[:-len_forecast]
    exog_test = exog_arr[-len_forecast:]
    # Data for lagged transformation
    train_lagged, predict_lagged = prepare_input_data(train_data, train_data, train_data, test_data, len_forecast, task)
    # Data for exog operation
    train_exog, predict_exog = prepare_input_data(exog_train, train_data, exog_test, test_data, len_forecast, task)

    train_dataset = MultiModalData({'lagged': train_lagged, 'exog_ts_data_source': train_exog})
    predict_dataset = MultiModalData({'lagged': predict_lagged, 'exog_ts_data_source': predict_exog})

    pipeline = get_complex_pipeline()

    start_time = timeit.default_timer()
    pipeline.fit(train_dataset)
    amount_of_seconds = timeit.default_timer() - start_time
    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train pipeline\n')

    predicted_values = pipeline.predict(predict_dataset)
    old_predicted = predicted_values.predict
    old_predicted = np.ravel(np.array(old_predicted))

    pipeline = get_complex_pipeline()
    new_predicted = make_forecast_with_tuning(pipeline, train_dataset, predict_dataset, task, cv_folds)
    new_predicted = np.ravel(np.array(new_predicted))
    test_data = np.ravel(test_data)

    count_errors(test_data, old_predicted, new_predicted)

    start_point = len(time_series) - len_forecast * 2
    comparsion_plot(col_name, ts, old_predicted, new_predicted, len(train_data), start=0)
    comparsion_plot(col_name, ts, old_predicted, new_predicted, len(train_data), start_point)


def make_forecast_with_tuning(orig_pipeline, train_input, predict_input, task, cv_folds):
    pipeline_tuner = PipelineTuner(orig_pipeline, task, iterations=15)
    tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_input, loss_function=mean_squared_error,
                                                  loss_params={'squared': False}, cv_folds=cv_folds,
                                                  validation_blocks=3)
    tuned_pipeline.fit(train_input)
    predicted_values = tuned_pipeline.predict(predict_input)
    new_predicted_values = predicted_values.predict

    return new_predicted_values


def get_complex_pipeline():
    node_lagged = PrimaryNode('lagged')
    node_exog = PrimaryNode('exog_ts_data_source')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged, node_exog])
    pipeline = Pipeline(node_ridge)
    return pipeline


def interpolate(df):
    temp_df = pd.DataFrame()
    count_columns = len(list(df.columns))
    new_str = np.zeros(count_columns)  # создаем вектор для новых значений
    new_str[:] = np.nan  # заполняем его nan
    depth = df.Depth[0]  # устанавливаем первое значение глубины в качестве начального
    depths_list = []  # создаем список "правильных" глубин для дальнейшего удаления остальных из датафрейма

    # обходим массив и создаем недостающие значения глубины (остальные - NaN)
    while depth <= df.Depth.values[-1]:
        depth = np.round(depth + 0.15, 2)
        depths_list.append(depth)
        if depth not in df.Depth.values:
            new_str[0] = depth  # добавляем новое значение для глубины, вставляем в наш временный датафрейм
            str_to_pandas = pd.DataFrame([new_str], columns=list(df.columns))
            temp_df = temp_df.append(str_to_pandas)

    print(len(temp_df), "новых значений добавлено")

    df = df.append(temp_df).sort_values(by=['Depth'])
    df = df.set_index('Depth')
    df = df.interpolate(method='index')
    df.reset_index(level=0, inplace=True)

    # после интерполяции удаляем "лишние" старые значения, чтобы остался ряд с шагом .15
    inx_list = []  # составим список индексов для дальнейшего удаления
    for i in range(1, len(df.Depth)):
        if df.Depth[i] not in depths_list:
            inx_list.append(i)

    df = df.drop(index=inx_list)
    plot_series(df[:100])
    # df.to_csv('kaggle/well_log_interpolated.csv', index=False)

    return df


if __name__ == "__main__":
    data = pd.read_excel("kaggle/well_log.xlsx", sheet_name=0)

    # убираем лишние столбцы
    data.drop(columns=['SXO', 'Dtsyn', 'Vpsyn', 'sw new', 'sw new%', 'PHI2', 'ΔVp (m/s)'], inplace=True)
    print(data)

    plot_series(data)
    plot_series(data[:100])

    corr = data.corr()  # рисуем корреляционную матрицу
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
    plt.show()

    data = interpolate(data)

    for col in list(data.columns):
        ts = np.array(data[col])
        run_experiment_with_tuning(ts, col, len_forecast=500, cv_folds=2)
