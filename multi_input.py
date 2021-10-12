from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import timeit
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.api.main import Fedot
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.api.api_utils.data_definition import array_to_input_data  # FEDOT ver. 0.4.1
from fedot.core.data.multi_modal import MultiModalData

from main import plot_series, interpolate, comparsion_plot

# Игнорирование возникающих предупреждений.
warnings.filterwarnings('ignore')


def run_experiment_with_tuning(time_series, col_name, len_forecast=250, cv_folds=None):
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(len_forecast))

    # мы хотим прогнозировать PHIE (пористость), возьмем параметры с наибольшей корреляцией
    features_to_use = ['SW', 'CGR', 'NPHI', 'FN', 'delta_vp', 'PHIE']  # важно! некоторые названия были изменены
    ts = np.array(data[col_name])
    train_dataset, predict_dataset, = prepare_multimodal_data(time_series, features_to_use, len_forecast)
    print('train_dataset', train_dataset)
    print('predict_dataset', predict_dataset)

    # Prepare parameters for algorithm launch
    composer_params = {'max_depth': 6,  # max depth of the pipeline
                       'max_arity': 4,  # max arity of the pipeline nodes
                       'pop_size': 20,  # population size for composer
                       'num_of_generations': 100,  # number of generations for composer
                       'timeout': 2,  # composing time (minutes)
                       # 'available_operations': # list of model names to use
                       'with_tuning': None,  # allow hyperparameters tuning for the model
                       'cv_folds': None,  # number of folds for cross-validation
                       'validation_blocks': None,  # number of validation blocks for time series forecasting
                       # 'initial_pipeline'  # initial assumption for composing
                       # 'genetic_scheme' # name of the genetic scheme
                       'preset': 'ultra_light',  # name of preset for model building (e.g. 'light', 'ultra-light')
                       'metric': 'rmse'}  #

    # Посмотрим на получившийся пайплайн при помощи параметра vis=True
    forecast, obtained_pipeline = multi_automl_fit_forecast(train_dataset, predict_dataset, composer_params,
                                                            ts, len_forecast, vis=True)

    mse_metric = mean_squared_error(ts[-len_forecast:], forecast, squared=False)
    mae_metric = mean_absolute_error(ts[-len_forecast:], forecast)
    print(f'MAE - {mae_metric:.2f}')
    print(f'RMSE - {mse_metric:.2f}')

    # Visualise predictions
    # plot_results(ts, forecast, len_forecast)
    len_train_data = len(ts) - len_forecast
    plt.plot(np.arange(0, len(ts)), ts, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(forecast)), forecast, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data], [min(ts), max(ts)], c='black', linewidth=2)
    plt.ylabel(col_name, fontsize=15)
    plt.xlabel('Depth', fontsize=15)
    plt.legend(fontsize=15, loc='upper left')
    plt.grid()
    plt.show()

    new_predicted = make_forecast_with_tuning(obtained_pipeline, train_dataset, predict_dataset,
                                              task, cv_folds, ts[:len(ts) - len_forecast])

    start_point = len(time_series) - len_forecast * 2
    comparsion_plot(obtained_pipeline, col_name, ts, forecast, new_predicted[0], len_train_data, start=0)
    comparsion_plot(obtained_pipeline, col_name, ts, forecast, new_predicted[0], len_train_data, start_point)


def make_forecast_with_tuning(pipeline, train_input, predict_input, task, cv_folds, target):
    print("************************make_forecast_with_tuning************************")
    data_part_transformation_func = partial(array_to_input_data, target_array=target, task=task)
    sources = dict((f'data_source_ts/{data_part_key}', data_part_transformation_func(features_array=data_part))
                   for (data_part_key, data_part) in train_input.items())
    train_input_data = MultiModalData(sources)

    pipeline_tuner = PipelineTuner(pipeline, task, iterations=30)
    new_pipeline = pipeline_tuner.tune_pipeline(input_data=train_input_data, loss_function=mean_squared_error,
                                            loss_params={'squared': False}, cv_folds=cv_folds, validation_blocks=3)
    new_pipeline.fit(train_input_data)
    new_pipeline.show()

    sources = dict((f'data_source_ts/{data_part_key}', data_part_transformation_func(features_array=data_part))
                   for (data_part_key, data_part) in predict_input.items())
    predict_input_data = MultiModalData(sources)

    predicted_values = new_pipeline.predict(predict_input_data)
    new_predicted_values = predicted_values.predict

    return new_predicted_values


def multi_automl_fit_forecast(train_input: dict, predict_input: dict, composer_params: dict, target: np.array,
                              forecast_length: int, vis=True):
    task_params = TsForecastingParams(forecast_length=forecast_length)
    model = Fedot(problem='ts_forecasting', composer_params=composer_params, task_params=task_params, verbose_level=4)
    obtained_pipeline = model.fit(features=train_input, target=target)

    if vis:
        obtained_pipeline.show()
        obtained_pipeline.print_structure()

    forecast = model.predict(features=predict_input)

    return forecast, obtained_pipeline


def prepare_multimodal_data(dataframe: pd.DataFrame, features: list, forecast_length: int):
    multi_modal_train = {}
    multi_modal_test = {}
    for feature in features:
        feature_ts = np.array(dataframe[feature])[:-forecast_length]

        # Will be the same
        multi_modal_train.update({feature: feature_ts})
        multi_modal_test.update({feature: feature_ts})

    return multi_modal_train, multi_modal_test


data = pd.read_excel("kaggle/well_log.xlsx", sheet_name=0, nrows=3000)
# убираем лишние столбцы
data.drop(columns=['SXO', 'Dtsyn', 'Vpsyn', 'sw new', 'sw new%', 'PHI2', 'ΔVp (m/s)'], inplace=True)

# plot_series(data)
# plot_series(data[:100])

corr = data.corr()  # рисуем корреляционную матрицу
sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
# plt.show()

data = interpolate(data)

# our columns - ['Depth', 'CALI', 'CGR', 'DT', 'ILD', 'NPHI', 'PEF', 'PHIE',
# 'RHOB', 'RT', 'SGR', 'SW', 'Φda', 'ΦN', 'ΦND', 'Vpreal', 'ΔVp']
# we need to rename some of them
dict_columns = {'Φda': 'FDA',
                'ΦN': 'FN',
                'ΦND': 'FND',
                'Vpreal': 'VP',
                'ΔVp': 'delta_vp'}
data.rename(columns=dict_columns, inplace=True)
print(data)  # let's see

# Запуск
run_experiment_with_tuning(data, 'PHIE', len_forecast=500, cv_folds=2)

# for col in list(data.columns):
#     ts = np.array(data[col])
#     run_experiment_with_tuning(data, col, len_forecast=500, cv_folds=2)
