import eqsig.single
import numpy as np
import glob
from matplotlib import pyplot as plt
from tqdm import tqdm


data_path_x = '../eq-data/FKSH17/data_all_x/*.csv'
data_path_y = '../eq-data/FKSH17/data_all_y/*.csv'
save_name = 'data_all.npz'
save_name_freq = 'data_all_freq.npz'
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


# Frequency date
training_example_freq = {}

def process_data(file_list, dt, periods, ncols):
    SA = []
    for file_path in file_list:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        sa_values = []
        for col_index in range(ncols):
            if ncols > 1:
                value = data[:, col_index]
            elif ncols == 1:
                value = data
            else:
                raise NotImplemented
            sa_period_results = []
            for period in periods:
                record = eqsig.AccSignal(value, dt)
                record.generate_response_spectrum(response_times=period)
                sa_period_results.append(record.s_a)
            sa_values.append(np.hstack(sa_period_results))
        SA.append(np.vstack(sa_values))
        response_spectrum = np.transpose(np.vstack(SA))

        # Viz
        # fig, axes = plt.subplots(3, 1)
        # periods_concat = np.concatenate(periods)
        # for i, ax in enumerate(axes):
        #     ax.plot(periods_concat, response_spectrum[:, i])
        # plt.show()

    return response_spectrum

# Process data
training_example_freq = {}
for i, (file_x, file_y) in tqdm(enumerate(zip(csv_files_x, csv_files_y))):
    Spectrum_x = process_data([file_x], dt, periods, ncols=3)
    Spectrum_y = process_data([file_y], dt, periods, ncols=1)
    periods_concat = np.concatenate(periods)
    training_example_freq[f"earthquake-{i}"] = {
        "id": (file_x, file_y), "x": Spectrum_x, "y": Spectrum_y, "periods": periods_concat}

    # Viz
    fig1, axes = plt.subplots(3, 1)
    periods_concat = np.concatenate(periods)
    for i, ax in enumerate(axes):
        ax.plot(periods_concat, Spectrum_x[:, i])

    fig2, ax = plt.subplots()
    ax.plot(periods_concat, Spectrum_y)
    plt.show()
a=1

np.savez_compressed(
    f"/work2/08264/baagee/frontera/site-response/eq-data/FKSH17/{save_name_freq}", **training_example_freq)
