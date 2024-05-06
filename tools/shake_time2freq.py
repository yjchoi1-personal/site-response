import pandas as pd
import os
import glob
import utils
import numpy as np



work_dir = '/work2/08264/baagee/frontera/site-response'
shake_timeseries_path = f'{work_dir}/data/datasets/shake_results/timeseries/'
shake_freq_path = f'{work_dir}/data/datasets/shake_results/spectrum/'
period_ranges = ((0.01, 0.1, 167), (0.1, 1, 167), (1, 10, 166))
periods = [np.linspace(start, end, num, endpoint=False) for start, end, num in period_ranges]
dt = 0.01

# Get a list of all files in the folder
file_names = os.listdir(shake_timeseries_path)

for file_name in file_names:
    shake_result_paths = glob.glob(
        f'{shake_timeseries_path}/{file_name}/*.csv')
    shake_result_paths = sorted(shake_result_paths, key=lambda path: path.split('/')[-1])

    if not os.path.exists(f'{shake_freq_path}/{file_name}'):
        os.makedirs(f'{shake_freq_path}/{file_name}', exist_ok=True)

    shake_result_names = os.listdir(f'{shake_timeseries_path}/{file_name}')
    shake_result_names = sorted([name.split('.')[0] for name in shake_result_names], key=lambda x: int(x.split('_')[1]))

    for shake_result_path, shake_result_name in zip(shake_result_paths, shake_result_names):
        shake_result_timeseries = pd.read_csv(shake_result_path, header=0).to_numpy()
        shake_result_timeseries = shake_result_timeseries.squeeze(-1)
        shake_result_spectrum = utils.time2freq(
            shake_result_timeseries, dt, periods)
        shake_result_spectrum = pd.DataFrame(shake_result_spectrum)

        shake_result_spectrum.to_csv(
            f'{shake_freq_path}/{file_name}/{shake_result_name}.csv', index=False)
