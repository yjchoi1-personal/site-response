import eqsig.single
import numpy as np
import glob
from matplotlib import pyplot as plt
from tqdm import tqdm


data_path_x = '../eq-data/FKSH17/data_all_x/*.csv'
data_path_y = '../eq-data/FKSH17/data_all_y/*.csv'
save_dir = "/work2/08264/baagee/frontera/site-response/eq-data/FKSH17/"
save_name_train = 'spectrum_train.npz'
save_name_test = 'spectrum_test.npz'
train_range = [0, 80]
test_range = [80, 100]
dt = 0.01  # time step of acceleration time series
period_ranges = ((0.01, 0.1, 167), (0.1, 1, 167), (1, 10, 166))  # for response spectrum


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

def process_data(file_list_x, file_list_y, dt, periods, visualize=False):
    n_feature_x = 3
    n_feature_y = 1

    # init variables for statistics
    cumulative_count = 0
    cumulative_sum = np.zeros(n_feature_x)
    cumulative_sumsq = np.zeros(n_feature_x)

    data_dict = {"data": {}, "statistics": {}}
    for i, (file_x, file_y) in tqdm(enumerate(zip(file_list_x, file_list_y))):
        # Load data
        data_x = np.loadtxt(file_x, delimiter=',', skiprows=1)  # shape=(12000, 3)
        data_y = np.loadtxt(file_y, delimiter=',', skiprows=1)  # shape=(12000, )

        # Processing for frequency domain representation
        Spectrum_x = calculate_spectrum([data_x], dt, periods, ncols=n_feature_x)
        Spectrum_y = calculate_spectrum([data_y], dt, periods, ncols=n_feature_y)

        # Get sum and sum squared for statistics
        cumulative_count += len(Spectrum_x)
        cumulative_sum += np.sum(Spectrum_x, axis=0)
        cumulative_sumsq += np.sum(Spectrum_x ** 2, axis=0)

        # Statistics for cumulative data
        cumulative_mean = cumulative_sum / cumulative_count
        cumulative_std = np.sqrt(
            cumulative_sumsq/cumulative_count - cumulative_mean**2)

        data_dict["data"][f"earthquake-{i}"] = {
            "id": (file_x, file_y),
            "x": Spectrum_x,
            "y": Spectrum_y,
        }

        if visualize:
            # Viz
            fig1, axes = plt.subplots(3, 1)
            periods_concat = np.concatenate(periods)
            for ax_id, ax in enumerate(axes):
                ax.plot(periods_concat, Spectrum_x[:, ax_id])

            fig2, ax = plt.subplots()
            ax.plot(periods_concat, Spectrum_y)
            plt.show()

    data_dict["statistics"] = {
        "feature_mean": cumulative_mean,
        "feature_std": cumulative_std
    }

    return data_dict


def calculate_spectrum(datasets, dt, periods, ncols):
    SA = []
    for data in datasets:
        sa_values = []
        for col_index in range(ncols):
            values = data[:, col_index] if ncols > 1 else data
            sa_period_results = []
            for period in periods:
                record = eqsig.AccSignal(values, dt)
                record.generate_response_spectrum(response_times=period)
                sa_period_results.append(record.s_a)
            sa_values.append(np.hstack(sa_period_results))
        SA.append(np.vstack(sa_values))
    response_spectrum = np.transpose(np.vstack(SA))
    return response_spectrum


# Process and save training and testing data
training_data = process_data(train_files_x, train_files_y, dt, periods)
np.savez_compressed(f'{save_dir}/{save_name_train}', **training_data)

testing_data = process_data(test_files_x, test_files_y, dt, periods)
np.savez_compressed(f'{save_dir}/{save_name_test}', **testing_data)

