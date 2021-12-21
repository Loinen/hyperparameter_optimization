import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeit
import seaborn as sns
from datetime import datetime, timedelta
import pathlib
import random

from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum


def loss_plot(data_array, title, path="results"):
    now_time = datetime.now()
    time = now_time.strftime("%d_%m_%Y_%H-%M-%S")
    print("loss: ", data_array)
    plt.plot(data_array)
    plt.title('Loss history')
    plt.ylabel('mse')
    plt.xlabel('iteration')
    plt.savefig('{path}/{title}_{now}'.format(path=path, title=title, now=time))
    plt.show()


def plot_series(df):
    df.plot(subplots=True, figsize=(15, 10))
    plt.xlabel("N")
    plt.legend(loc='best')
    plt.xticks(rotation='vertical')
    plt.show()


def comparsion_plot(col_name, ts, old_predicted, new_predicted, train_len, start=0, path="results",
                    title="Comparsion plot"):
    time_dt = datetime.now()
    time = time_dt.strftime("%d_%m_%Y_%H-%M-%S")

    plt.plot(range(start, len(ts)), ts[start:], label='Actual time series')
    plt.plot(range(train_len, len(ts)), old_predicted, label='Forecast before tuning', linestyle='--', color='c')
    plt.plot(range(train_len, len(ts)), new_predicted, label='Forecast after tuning', color='g')
    plt.title(title)
    plt.ylabel(col_name)
    plt.xlabel("N")
    plt.legend()
    plt.grid()
    plt.savefig('{path}/{title}_{now}'.format(path=path, title=title, now=time))
    plt.show()


def count_errors(test_data, old_predicted, new_predicted):
    mse_before = mean_squared_error(test_data, old_predicted, squared=False)
    mae_before = mean_absolute_error(test_data, old_predicted)
    print(f'MSE before tuning - {mse_before:.4f}')
    print(f'MAE before tuning - {mae_before:.4f}\n')

    mse_after = mean_squared_error(test_data, new_predicted, squared=False)
    mae_after = mean_absolute_error(test_data, new_predicted)
    print(f'MSE after tuning - {mse_after:.4f}')
    print(f'MAE after tuning - {mae_after:.4f}\n')

    return round(mse_before, 4), round(mae_before, 4), round(mse_after, 4), round(mae_after, 4)


def prepare_input_data(features_train_data, target_train_data, features_test_data, target_test, len_forecast, task):
    train_input = InputData(idx=np.arange(0, len(features_train_data)), features=features_train_data,
                            target=target_train_data, task=task, data_type=DataTypesEnum.ts)
    start_forecast = len(features_train_data)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast), features=features_test_data,
                              target=target_test, task=task, data_type=DataTypesEnum.ts)

    return train_input, predict_input


def run_experiment_with_tuning(time_series, col_name, exog_variable, len_forecast=250, cv_folds=None):
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(len_forecast))

    start_exp_time = datetime.now()
    start_time = start_exp_time.strftime("%d_%m_%Y_%H-%M-%S")
    start_path = 'results/{col_name}_exog/{start_time}'.format(start_time=start_time, col_name=col_name)
    pathlib.Path(start_path).mkdir(parents=True, exist_ok=True)

    # divide on train and test
    train_input, predict_input = train_test_data_setup(InputData(idx=range(len(time_series)), features=time_series,
                                                                 target=time_series, task=task,
                                                                 data_type=DataTypesEnum.ts))
    predict_input_exog = InputData(idx=np.arange(len(exog_variable)), features=exog_variable, target=time_series,
                                   task=task, data_type=DataTypesEnum.ts)
    train_input_exog, _ = train_test_data_setup(predict_input_exog)
    # train_data = time_series[:-len_forecast]
    # test_data = time_series[-len_forecast:]
    # exog_train = exog_variable[:-len_forecast]
    # exog_test = exog_variable[-len_forecast:]
    # train_lagged, predict_lagged = prepare_input_data(train_data, train_data, train_data, test_data, len_forecast, task)
    # train_exog, predict_exog = prepare_input_data(exog_train, train_data, exog_test, test_data, len_forecast, task)
    # train_dataset = MultiModalData({'data_source_ts/1': train_lagged, 'exog_ts': train_exog})
    # predict_dataset = MultiModalData({'data_source_ts/1': predict_lagged, 'exog_ts': predict_exog})

    pipeline = get_complex_pipeline()
    pipeline.show()

    second_node_name = 'exog_ts'
    train_dataset = MultiModalData({
        'data_source_ts/1': train_input,
        second_node_name: train_input_exog,
    })

    predict_dataset = MultiModalData({
        'data_source_ts/1': predict_input,
        second_node_name: predict_input_exog,
    })

    old_predicted, new_predicted, tune_time, tuned_pipe = \
        make_forecast_with_tuning(pipeline, train_dataset, predict_dataset, task, cv_folds)
    old_predicted = np.ravel(np.array(old_predicted))
    new_predicted = np.ravel(np.array(new_predicted))

    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]
    test_data = np.ravel(test_data)

    # plot losses
    outfile = "C:/Users/User/PycharmProjects/hyperparameter_optimization/results/loss_hyperopt/losses.npy"
    losses = np.load(outfile)
    loss_plot(losses, "loss_pipeline_tuner", start_path)

    mse_init, mae_init, mse_tuning, mae_tuning = count_errors(test_data, old_predicted, new_predicted)

    start_point = len(time_series) - len_forecast * 2
    comparsion_plot(col_name, ts, old_predicted, new_predicted, len(train_data), 0, start_path,
                    "Init pipeline vs tuned (green) pipeline start=0")
    comparsion_plot(col_name, ts, old_predicted, new_predicted, len(train_data), start_point, start_path,
                    "Init pipeline vs tuned (green) pipeline")

    # random search
    start_time = timeit.default_timer()
    results_rs, history = baseline(time_series, exog_variable, len_forecast, train_data)
    amount_of_seconds = timeit.default_timer() - start_time
    print(f'\nIt takes {amount_of_seconds:.2f} seconds to tune RS pipeline\n')
    loss_plot(history, "loss_RS", start_path)

    # get rs pipe and save rs pipe params
    rs_pipeline = get_complex_pipeline(results_rs)
    rs_pipeline.print_structure()
    rs_params = 'params: '
    for node in rs_pipeline.nodes:
        rs_params += str(f"\n{node.operation.operation_type} - {node.custom_params}")
    print('rs_params', rs_params)

    rs_pipeline.fit_from_scratch(train_dataset)
    predicted_values = rs_pipeline.predict(predict_dataset)
    rs_predicted_values = predicted_values.predict
    rs_predicted_values = np.ravel(np.array(rs_predicted_values))

    mse_init, mae_init, mse_rs, mae_rs = count_errors(test_data, old_predicted, rs_predicted_values)
    start_point = len(time_series) - len_forecast * 2
    comparsion_plot(col_name, ts, old_predicted, rs_predicted_values, len(train_data), start_point, start_path,
                    "Init pipeline vs RS (green) pipeline")
    comparsion_plot(col_name, ts, new_predicted, rs_predicted_values, len(train_data), start_point, start_path,
                    "Tuned pipeline vs RS (green) pipeline")

    # create files with resulting metrics and params
    now_dt = datetime.now()
    now = now_dt.strftime("%d_%m_%Y_%H-%M-%S")
    errors_str = ' '.join(str(x) for x in [mse_init, mae_init, mse_tuning, mae_tuning])
    errors_rs = ' '.join(str(x) for x in [mse_rs, mae_rs])

    # params for tuned pipeline
    tuned_params = 'params: '
    for node in tuned_pipe.nodes:
        tuned_params += str(f"\n{node.operation.operation_type} - {node.custom_params}")
    print('tuned_params', tuned_params)

    with open('{errors_path}/errors_{now}.txt'.format(errors_path=start_path, now=now), 'w') as f:
        f.write('mse_init, mae_init, mse_tuning, mae_tuning \n' + errors_str + '\ntune_time \n' + str(tune_time) +
                '\n mse_rs, mae_rs \n' + errors_rs + '\nrs_tune_time \n' + str(amount_of_seconds) +
                '\n\nrs_best_params \n' + str(rs_params) + '\n\ntuned_best_params \n' + str(tuned_params))

    res = [mse_init, mae_init, mse_tuning, mae_tuning, mse_rs, mae_rs, round(tune_time, 2), round(amount_of_seconds, 2)]
    res_df = pd.DataFrame([res], columns=['mse_init', 'mae_init', 'mse_tuning', 'mae_tuning', 'mse_rs', 'mae_rs',
                                          'tune_time', 'rs_time'])
    print(res_df)
    res_df.to_excel('{errors_path}/errors_xlsx_{now}.xlsx'.format(errors_path=start_path, now=now), index=False)
    return res_df


def make_forecast_with_tuning(orig_pipeline, train_input, predict_input, task, cv_folds):
    orig_pipeline.fit(train_input)
    predicted_values = orig_pipeline.predict(predict_input)
    old_predicted_values = predicted_values.predict

    pipeline_tuner = PipelineTuner(orig_pipeline, task, iterations=20, timeout=timedelta(minutes=5))

    start_time = timeit.default_timer()
    tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_input, loss_function=mean_squared_error,
                                                  loss_params={'squared': False}, cv_folds=None,
                                                  validation_blocks=None)  # кросс-валидация не работает для multi_modal
    amount_of_seconds = timeit.default_timer() - start_time
    print(f'\nIt takes {amount_of_seconds:.2f} seconds to tune pipeline\n')

    tuned_pipeline.fit_from_scratch(train_input)
    predicted_values = tuned_pipeline.predict(predict_input)
    new_predicted_values = predicted_values.predict

    return old_predicted_values, new_predicted_values, amount_of_seconds, tuned_pipeline


def get_complex_pipeline(start_params=None):
    """
    Pipeline looking like this
    source - smoothing - lagged - ridge \
                                         \
                               exog_node - ridge -> final forecast
                                        /
              source - lagged - ridge /
    """
    node_source = PrimaryNode('data_source_ts/1')
    node_source2 = PrimaryNode('data_source_ts/1')
    node_exog = PrimaryNode('exog_ts')

    # Second level
    node_smoothing = SecondaryNode('smoothing', nodes_from=[node_source])

    if start_params is not None:
        print('start params:', start_params)
        node_smoothing.custom_params = {'window_size': start_params['window_size_smooth']}

    # Third level
    node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_smoothing])
    node_lagged_2 = SecondaryNode('lagged', nodes_from=[node_source2])

    if start_params is not None:
        node_lagged_1.custom_params = {'window_size': start_params['window_size']}
        node_lagged_2.custom_params = {'window_size': start_params['window_size2']}

    # Fourth level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    if start_params is not None:
        node_ridge_1.custom_params = {'alpha': start_params['alpha']}
        node_ridge_2.custom_params = {'alpha': start_params['alpha2']}

    # Fifth level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_exog, node_ridge_2])
    if start_params is not None:
        node_final.custom_params = {'alpha': start_params['alpha3']}

    pipeline = Pipeline(node_final)
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


def baseline(ts_var, exog_var, len_forecast, train_ts):
    min_mse = 9999
    num_iters = 20
    real_data = np.ravel(train_ts[:-len_forecast])
    ts_var = ts_var[:-len_forecast]
    exog_var = exog_var[:-len_forecast]
    best_params = None
    mse_array = []
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(len_forecast))

    train_input, predict_input = train_test_data_setup(InputData(idx=range(len(ts_var)), features=ts_var,
                                                                 target=ts_var, task=task, data_type=DataTypesEnum.ts))
    predict_input_exog = InputData(idx=np.arange(len(exog_var)), features=exog_var, target=ts_var,
                                   task=task, data_type=DataTypesEnum.ts)
    train_input_exog, _ = train_test_data_setup(predict_input_exog)
    second_node_name = 'exog_ts'
    train_data = MultiModalData({
        'data_source_ts/1': train_input,
        second_node_name: train_input_exog,
    })

    predict_dataset = MultiModalData({
        'data_source_ts/1': predict_input,
        second_node_name: predict_input_exog,
    })

    for iteration in range(num_iters):
        print('iteration', iteration, '/', num_iters)
        alpha_range = np.arange(0.01, 10, 0.01)
        lagged = np.arange(5, 500, 1)
        smooth = np.arange(2, 20, 1)
        params = dict(alpha=random.choice(alpha_range), alpha2=random.choice(alpha_range),
                      window_size=random.choice(lagged), window_size_smooth=random.choice(smooth),
                      alpha3=random.choice(alpha_range), window_size2=random.choice(lagged))

        pipeline = get_complex_pipeline(params)
        pipeline.fit_from_scratch(train_data)
        predicted_values = pipeline.predict(predict_dataset)
        predicted_values = predicted_values.predict
        predicted_values = np.ravel(np.array(predicted_values))
        mse = mean_squared_error(real_data[len(real_data) - 500:], predicted_values, squared=False)
        mse_array.append(mse)

        if mse < min_mse:
            min_mse = mse
            best_params = params

    print("best params is: ", best_params)
    print("min_mse is: ", min_mse)

    return best_params, mse_array


if __name__ == "__main__":
    data = pd.read_excel("kaggle/well_log.xlsx", sheet_name=0, nrows=3000)

    # убираем лишние столбцы
    data.drop(columns=['SXO', 'Dtsyn', 'Vpsyn', 'sw new', 'sw new%', 'PHI2', 'ΔVp (m/s)', 'ΦN'], inplace=True)
    plot_series(data)
    plot_series(data[:100])

    corr = data.corr()  # рисуем корреляционную матрицу
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
    plt.show()

    data = interpolate(data)
    dict_columns = {'Φda': 'FDA',
                    'ΦND': 'FND',
                    'Vpreal': 'VP'}
    data.rename(columns=dict_columns, inplace=True)

    iterations = 10

    now_date = datetime.now()
    now = now_date.strftime("%d_%m_%Y_%H-%M-%S")
    result_errors_path = 'results/errors_exog'
    pathlib.Path(result_errors_path).mkdir(parents=True, exist_ok=True)
    loss_npy_path = 'results/loss_hyperopt'
    pathlib.Path(loss_npy_path).mkdir(parents=True, exist_ok=True)
    # for saving losses you should add this two strings to hyperopt/fmin.py after 553 str
    # outfile = "C:/Users/User/PycharmProjects/hyperparameter_optimization/results/loss_hyperopt/losses.npy"
    # np.save(outfile, trials.losses())

    for col in ['NPHI', 'DT']:  # ['DT', 'SW', 'CGR', 'NPHI', 'PHIE']
        # 'NPHI', 'DT' - RHOB          -
        # 'SW' - 'PHIE'                -
        # 'CGR' - SGR                  -
        # 'PHIE' - 'SW'                -
        resulting_df = pd.DataFrame()
        ts = np.array(data[col])
        exog_ts = np.array(data['RHOB'])
        for i in range(iterations):
            temp_result = run_experiment_with_tuning(ts, col, exog_ts, len_forecast=500, cv_folds=2)
            resulting_df = resulting_df.append(temp_result, ignore_index=True)
        print(resulting_df)
        print(resulting_df.describe())

        resulting_df.to_excel('{path}/{col}_err_{now}_for_{i}_iters.xlsx'.format(now=now, i=iterations, col=col,
                                                                                 path=result_errors_path), index=False)
        resulting_df.describe().to_excel(
            '{path}/{col}_err_{now}_for_{i}_iters_describe.xlsx'.format(now=now, i=iterations, col=col,
                                                                        path=result_errors_path), index=False)
