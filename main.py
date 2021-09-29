import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import timeit

from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

# Игнорирование возникающих предупреждений.
warnings.filterwarnings('ignore')


def plot_series(df):
    df.plot(subplots=True, figsize=(15, 10))
    plt.xlabel("N")
    plt.legend(loc='best')
    plt.xticks(rotation='vertical')
    plt.show()


def run_experiment_with_tuning(time_series, col_name, with_ar_pipeline=False, len_forecast=250, cv_folds=None):
    """ Function with example how time series forecasting can be made
    :param cv_folds: number of folds for validation
    :param time_series: time series for prediction
    :param with_ar_pipeline: is it needed to use pipeline with AR model or not
    :param len_forecast: forecast length
    """

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast, train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)
    # Get graph with several models and with arima pipeline
    if with_ar_pipeline:
        pipeline = get_ar_pipeline()
    else:
        pipeline = get_complex_pipeline()

    old_predicted, new_predicted = make_forecast_with_tuning(pipeline, train_input, predict_input,
                                                             task, cv_folds)

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

    pipeline.print_structure()
    plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
    plt.plot(range(len(train_data), len(time_series)), old_predicted, label='Forecast before tuning')
    plt.plot(range(len(train_data), len(time_series)), new_predicted, label='Forecast after tuning')
    plt.ylabel(col_name)
    plt.xlabel("N")
    plt.legend()
    plt.grid()
    plt.show()

    start_point = len(time_series) - len_forecast * 2
    plt.plot(range(start_point, len(time_series)), time_series[start_point:], label='Actual time series')
    plt.plot(range(len(train_data), len(time_series)), old_predicted, label='Forecast before tuning')
    plt.plot(range(len(train_data), len(time_series)), new_predicted, label='Forecast after tuning')
    plt.ylabel(col_name)
    plt.xlabel("N")
    plt.legend()
    plt.grid()
    plt.show()


def make_forecast_with_tuning(pipeline, train_input, predict_input, task, cv_folds):
    """
    Function for predicting values in a time series
    :param pipeline: TsForecastingPipeline object
    :param train_input: InputData for fit
    :param predict_input: InputData for predict
    :param task: Ts_forecasting task
    :param cv_folds: number of folds for validation
    :return predicted_values: numpy array, forecast of model
    """

    # Fit it
    start_time = timeit.default_timer()
    pipeline.fit_from_scratch(train_input)
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train pipeline\n')

    # Predict
    predicted_values = pipeline.predict(predict_input)
    old_predicted_values = predicted_values.predict

    pipeline_tuner = PipelineTuner(pipeline, task, iterations=10)
    pipeline = pipeline_tuner.tune_pipeline(input_data=train_input,
                                            loss_function=mean_squared_error,
                                            loss_params={'squared': False},
                                            cv_folds=cv_folds,
                                            validation_blocks=3)

    # Fit pipeline on the entire train data
    pipeline.fit_from_scratch(train_input)
    # Predict
    predicted_values = pipeline.predict(predict_input)
    new_predicted_values = predicted_values.predict

    return old_predicted_values, new_predicted_values


def get_complex_pipeline():
    """
    Pipeline looking like this
    smoothing - lagged - ridge \
                                \
                                 ridge -> final forecast
                                /
                lagged - ridge /
    """

    # First level
    node_smoothing = PrimaryNode('smoothing')

    # Second level
    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_smoothing])
    node_lagged_2 = PrimaryNode('lagged')

    # Third level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    # Fourth level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)

    return pipeline


def get_ar_pipeline():  # Function return graph with AR model
    node_ar = PrimaryNode('ar')
    pipeline = Pipeline(node_ar)
    return pipeline


def prepare_input_data(len_forecast, train_data_features, train_data_target, test_data_features):
    """ Return prepared data for fit and predict
    :param len_forecast: forecast length
    :param train_data_features: time series which can be used as predictors for train
    :param train_data_target: time series which can be used as target for train
    :param test_data_features: time series which can be used as predictors for prediction
    :return train_input: Input Data for fit
    :return predict_input: Input Data for predict
    :return task: Time series forecasting task with parameters
    """

    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data_features)),
                            features=train_data_features, target=train_data_target,
                            task=task, data_type=DataTypesEnum.ts)

    start_forecast = len(train_data_features)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(0, end_forecast),
                              features=np.concatenate([train_data_features, test_data_features]),
                              target=None, task=task, data_type=DataTypesEnum.ts)

    return train_input, predict_input, task


data = pd.read_excel("kaggle/well_log.xlsx", sheet_name=0)
# убираем лишние столбцы
data.drop(columns=['SXO', 'Dtsyn', 'Vpsyn', 'sw new', 'sw new%', 'PHI2', 'ΔVp (m/s)'], inplace=True)
print(data)

plot_series(data[:100])

# рисуем корреляционную матрицу
# corr = data.corr()
# sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
# plt.show()

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
print(data)

data = data.interpolate(method='index')
plot_series(data[:100])

# после интерполяции удаляем "лишние" старые значения, чтобы остался ряд с шагом .15
inx_list = []
for i in range(1, len(data.Depth)):
    if data.Depth[i] not in depths_list:
        inx_list.append(i)

data = data.drop(index=inx_list)
data = data.drop(columns=["index"])

plot_series(data[:100])

print(data)
print(data[:25])

# data.to_csv('kaggle/well_log_interpolated.csv', index=False)

# Запуск.
for col in list(data.columns):
    ts = np.array(data[col])
    run_experiment_with_tuning(ts, col, with_ar_pipeline=False, len_forecast=300, cv_folds=2)
