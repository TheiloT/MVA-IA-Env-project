from darts.models import LinearRegressionModel
from tqdm.auto import tqdm

from waste_prediction.experiments.helpers import run

DATASET_LIST = [
    # ('boralasgamuwa_uc_2012-2018', '2016-05-01 00:00:00'),
    ('moratuwa_mc_2014-2018', '2017-11-01 00:00:00', '2018-06-01 00:00:00'),
    # ('dehiwala_mc_2012-2018', '2015-02-01 00:00:00'),
    # ('open_source_ballarat_daily_waste_2000_jul_2015_mar', '2008-09-01 00:00:00'),
    # ('open_source_austin_daily_waste_2003_jan_2021_jul', '2015-06-01 00:00:00')
]


def generate_model_name(params):
    model_name = 'daily_lr_weekdays-{}_diff-{}_lags-{}'.format(
        params['only_weekdays'],
        params['is_differenced'],
        params['n_lags']
    )

    return model_name


def generate_model(params):
    model = LinearRegressionModel(
        lags=params['n_lags'],
    )

    return model


def run_tests():
    # n_lags_list = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    n_lags_list = [150]
    n_iter = len(n_lags_list)

    for dataset_name, test_calibration_split_before, empirical_coverage_split_before in DATASET_LIST:
        with tqdm(total=n_iter) as pbar:
            for n_lags in n_lags_list:
                params = {
                    'dataset_name': dataset_name,
                    'test_calibration_split_before': test_calibration_split_before,
                    'empirical_coverage_split_before': empirical_coverage_split_before,
                    'only_weekdays': False,
                    'is_differenced': False,

                    'n_lags': n_lags
                }
                try:
                    run(params, generate_model_name, generate_model)
                except:
                    print('Error: {}'.format(generate_model_name(params)))
                pbar.update(1)


if __name__ == '__main__':
    run_tests()
