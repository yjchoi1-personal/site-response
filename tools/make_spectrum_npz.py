import eqsig.single
import numpy as np
import glob
import argparse
import pandas as pd
import os
from matplotlib import pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default="FKSH19", type=str, help="Earthquake data name")
args = parser.parse_args()

work_dir = "/work2/08264/baagee/frontera/site-response/"
data_path_x = f'{work_dir}/data/datasets/{args.dataset_name}/data_all_x/*.csv'
data_path_y = f'{work_dir}/data/datasets/{args.dataset_name}/data_all_y/*.csv'
save_dir = f"{work_dir}/data/datasets/{args.dataset_name}/"
save_name_train = 'spectrum_train.npz'
save_name_test = 'spectrum_test.npz'
train_range = [0, 80]
test_range = [80, 100]
dt = 0.01  # time step of acceleration time series
period_ranges = ((0.01, 0.1, 167), (0.1, 1, 167), (1, 10, 166))  # for response spectrum
n_feature_x = 3
period_feature = False
visualize = True
save_csv = True


csv_files_x = glob.glob(data_path_x)
csv_files_y = glob.glob(data_path_y)
# Sort
csv_files_x = sorted(csv_files_x, key=lambda path: path.split('/')[-1])
csv_files_y = sorted(csv_files_y, key=lambda path: path.split('/')[-1])

if len(csv_files_x) != len(csv_files_y):
    raise ValueError("x and y should have the same length")
else:
    n_datasets = len(csv_files_x)

# Inputs for response spectrum
periods = [np.linspace(start, end, num, endpoint=False) for start, end, num in period_ranges]

# Split data into training and testing
train_files_x = csv_files_x[train_range[0]:train_range[1]]
train_files_y = csv_files_y[train_range[0]:train_range[1]]
test_files_x = csv_files_x[test_range[0]:test_range[1]]
test_files_y = csv_files_y[test_range[0]:test_range[1]]

def process_data(
        file_list_x,
        file_list_y,
        dt,
        periods,
        n_feature_x,
        tag,
        period_feature=False,
        visualize=False,
        save_csv=False
):

    print("-----------------------")
    print(f"Filename: {file_list_x}")
    if not file_list_x:
        raise ValueError("There are no data in the folder")

    if visualize:
        vis_save_dir = f"{work_dir}/data/datasets/{args.dataset_name}/{tag}_vis/"
        if not os.path.exists(vis_save_dir):
            os.makedirs(vis_save_dir, exist_ok=True)
    if save_csv:
        csv_save_dir = f"{work_dir}/data/datasets/{args.dataset_name}/{tag}_csv/"
        if not os.path.exists(csv_save_dir):
            os.makedirs(csv_save_dir, exist_ok=True)

    # init variables for statistics
    cumulative_count = 0
    cumulative_sum = np.zeros(n_feature_x)
    cumulative_sumsq = np.zeros(n_feature_x)
    cumulative_max = np.zeros(n_feature_x)
    cumulative_min = np.full(n_feature_x, 1e7)

    # spectrum_x_save = []
    # spectrum_y_save = []

    data_dict = {"data": {}, "statistics": {}}

    # Iterate data
    for i, (file_x, file_y) in tqdm(enumerate(zip(file_list_x, file_list_y))):
        # Load data
        data_x = np.loadtxt(file_x, delimiter=',', skiprows=1)  # shape=(12000, 3)
        data_y = np.loadtxt(file_y, delimiter=',', skiprows=1)  # shape=(12000, )
        # If data_y has 3 features, only select the first column which is accel.
        if data_y.ndim == 2 and data_y.shape[1] == 3:
            data_y = data_y[:, 0]

        # Processing for frequency domain representation
        Spectrum_x = calculate_spectrum(
            data_x, dt, periods, ncols=data_x.shape[-1], period_feature=period_feature)
        Spectrum_y = calculate_spectrum(
            data_y, dt, periods, ncols=1, period_feature=False)

        # spectrum_x_save.append(Spectrum_x)
        # spectrum_y_save.append(Spectrum_y)

        # Get sum and sum squared for statistics
        cumulative_count += len(Spectrum_x)
        cumulative_sum += np.sum(Spectrum_x, axis=0)
        cumulative_sumsq += np.sum(Spectrum_x ** 2, axis=0)

        # Statistics for cumulative data
        cumulative_mean = cumulative_sum / cumulative_count
        cumulative_std = np.sqrt(
            cumulative_sumsq/cumulative_count - cumulative_mean**2)
        current_max = Spectrum_x.max(axis=0)
        current_min = Spectrum_x.min(axis=0)
        cumulative_max = np.where(current_max > cumulative_max, current_max, cumulative_max)
        cumulative_min = np.where(current_min < cumulative_min, current_min, cumulative_min)

        # Split the string by '/'
        site_name_x = file_x.split('/')[-1].replace('.csv', '')
        site_name_y = file_y.split('/')[-1].replace('.csv', '')

        data_dict["data"][f"earthquake-{i}"] = {
            "id": [site_name_x, site_name_y],
            "x": Spectrum_x,
            "y": Spectrum_y,
        }

        # Visualize current data (time series & spectrum)
        if visualize:

            spectrum_ylabel = ["SA", "SV", "SD"]
            timeseries_ylabel = ["a", "v", "d"]

            # Viz x data
            fig1, axes = plt.subplots(3, 2)
            periods_concat = np.concatenate(periods)
            # Time series
            for ax_id, ax in enumerate(axes[:, 0]):
                ax.plot(data_x[:, ax_id])
                ax.set_xlabel("Period (sec)")
                ax.set_ylabel(timeseries_ylabel[ax_id])
            # Spectrum
            for ax_id, ax in enumerate(axes[:, 1]):
                ax.plot(periods_concat, Spectrum_x[:, ax_id])
                ax.set_xlabel("Period (sec)")
                ax.set_ylabel(spectrum_ylabel[ax_id])
                ax.set_xscale('log')
            plt.tight_layout()
            plt.savefig(f"{vis_save_dir}/dataset{i}_x-{site_name_x}.png")

            # Viz y data
            fig2, ax = plt.subplots(1, 2, figsize=(9, 3))
            # Time series
            ax[0].plot(data_y)
            ax[0].set_xlabel("Period (sec)")
            ax[0].set_ylabel("A (g)")
            # Spectrum
            ax[1].plot(periods_concat, Spectrum_y)
            ax[1].set_xlabel("Period (sec)")
            ax[1].set_ylabel("SA (g)")
            ax[1].set_xscale('log')
            plt.tight_layout()
            plt.savefig(f"{vis_save_dir}/dataset{i}_y-{site_name_y}.png")

        if save_csv:
            # Add period column
            df_spectrum_x = np.concatenate((periods_concat.reshape(-1, 1), Spectrum_x), axis=1)
            df_spectrum_y = np.concatenate((periods_concat.reshape(-1, 1), Spectrum_y), axis=1)
            # Make it to dataframe
            df_spectrum_x = pd.DataFrame(
                df_spectrum_x, columns=["Period", "sa", "sv", "sd"])
            df_spectrum_y = pd.DataFrame(
                df_spectrum_y, columns=["Period", "sa"])
            # Save csv
            df_spectrum_x.to_csv(f'{csv_save_dir}/dataset{i}_x-{site_name_y}.csv', index=False)
            df_spectrum_y.to_csv(f'{csv_save_dir}/dataset{i}_y-{site_name_y}.csv', index=False)

    data_dict["statistics"] = {
        "feature_mean": cumulative_mean,
        "feature_std": cumulative_std,
        "feature_max": cumulative_max,
        "feature_min": cumulative_min,
    }

    # concat_x = np.concatenate(spectrum_x_save)
    # concat_y = np.concatenate(spectrum_y_save)
    # concat_x_mean = concat_x.mean(0)
    # concat_x_std = concat_x.std(0)

    print(data_dict["statistics"])
    # print(f"concat x mean: {concat_x_mean}")
    # print(f"concat x std: {concat_x_std}")
    # print(f"concat x max: {concat_x.max(0)}")
    # print(f"concat x min: {concat_x.min(0)}")

    return data_dict


def calculate_spectrum(data, dt, periods, ncols, period_feature=False):
    sa_values = []
    for col_index in range(ncols):
        values = data[:, col_index] if ncols > 1 else data
        sa_period_results = []
        for period in periods:
            record = eqsig.AccSignal(values, dt)
            record.generate_response_spectrum(response_times=period)
            sa_period_results.append(record.s_a)
        sa_values.append(np.hstack(sa_period_results))
    if period_feature:
        periods_concat = np.concatenate(periods)
        # Normalize
        periods_concat = periods_concat / periods_concat.max()
        sa_values.append(periods_concat)
    response_spectrum = np.transpose(np.vstack(sa_values))
    return response_spectrum


# Process and save training and testing data
training_data = process_data(
    train_files_x, train_files_y, dt, periods, n_feature_x,
    tag="train", period_feature=period_feature,
    visualize=visualize, save_csv=save_csv)
np.savez_compressed(f'{save_dir}/{save_name_train}', **training_data)

testing_data = process_data(
    test_files_x, test_files_y, dt, periods, n_feature_x,
    tag="test", period_feature=period_feature,
    visualize=visualize, save_csv=save_csv)
np.savez_compressed(f'{save_dir}/{save_name_test}', **testing_data)

