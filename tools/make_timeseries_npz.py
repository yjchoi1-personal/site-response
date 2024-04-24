import numpy as np
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt


# Replace 'path/to/your/csv_files/*.csv' with the actual path and pattern matching your CSV files
data_path_x = '../eq-data/FKSH17/data_all_x/*.csv'
data_path_y = '../eq-data/FKSH17/data_all_y/*.csv'
save_dir = "/work2/08264/baagee/frontera/site-response/eq-data/FKSH17/"
save_name_train = 'timeseries_train.npz'
save_name_test = 'timeseries_test.npz'
train_range = [0, 80]
test_range = [80, 100]


# Get file names
csv_files_x = glob.glob(data_path_x)
csv_files_y = glob.glob(data_path_y)
# Sort
csv_files_x = sorted(csv_files_x, key=lambda path: path.split('/')[-1])
csv_files_y = sorted(csv_files_y, key=lambda path: path.split('/')[-1])
# Check the length
if len(csv_files_x) != len(csv_files_y):
    raise ValueError("x and y should have the same length")
else:
    n_datasets = len(csv_files_x)

# Split files into training and testing
train_files_x = csv_files_x[train_range[0]:train_range[1]]
train_files_y = csv_files_y[train_range[0]:train_range[1]]
test_files_x = csv_files_x[test_range[0]:test_range[1]]
test_files_y = csv_files_y[test_range[0]:test_range[1]]

# Initialize a list to hold the 2D arrays from each CSV file
training_examples = {}
testing_examples = {}

# Loop through each file, read the data into a 2D numpy array, and append to the list
for i, (file_x, file_y) in enumerate(zip(train_files_x, train_files_y)):
    data_x = np.loadtxt(file_x, delimiter=',', skiprows=1)
    data_y = np.loadtxt(file_y, delimiter=',', skiprows=1)
    training_examples[f"earthquake-{i}"] = {
        "id": (csv_files_x[i], csv_files_y[i]), "x": data_x, "y": data_y}

for i, (file_x, file_y) in enumerate(zip(test_files_x, test_files_y)):
    data_x = np.loadtxt(file_x, delimiter=',', skiprows=1)
    data_y = np.loadtxt(file_y, delimiter=',', skiprows=1)
    testing_examples[f"earthquake-{i}"] = {
        "id": (csv_files_x[i], csv_files_y[i]), "x": data_x, "y": data_y}

np.savez_compressed(f"{save_dir}/{save_name_train}", **training_examples)
np.savez_compressed(f"{save_dir}/{save_name_test}", **testing_examples)


