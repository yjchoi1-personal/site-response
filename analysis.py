import os.path
import matplotlib as mpl
import numpy as np
import pickle
import pandas as pd
import json
import utils
from matplotlib import pyplot as plt

a = mpl.rcParams
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams.update({'font.size': 10})
fig_format = 'png'

# plot inputs
config = {
    "cnn":
        {
            "color": 'blue',
            "linestyle": '-'
        },
    "lstm":
        {
            "color": 'red',
            "linestyle": '--'
        },
    "transformer":
        {
            "color": 'green',
            "linestyle": '-.'
        },
    "shake":
        {
            "color": 'silver',
            "linestyle": '-'
        }
}


period_ranges = ((0.01, 0.1, 167), (0.1, 1, 167), (1, 10, 166))
periods = [np.linspace(start, end, num, endpoint=False) for start, end, num in period_ranges]


def site_analysis(site):
    shake_data_path = f'data/outputs/{site}-shake/'
    save_dir = f'data/analysis/{site}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # del config['transformer']
    model_ids = list(config.keys())

    # Pick one arbitrary model and get site ids
    results_path = f'data/outputs/{site}-{model_ids[0]}/'
    # Load any results from picked model
    with open(f'{results_path}/results.pkl', 'rb') as file:
        loaded_dict = pickle.load(file)
    # Get site_ids to analyze and shake result file ids
    site_ids = loaded_dict.keys()
    shake_ids = [value['file_names'][1] for value in loaded_dict.values()]

    # Get shake results and compute MSE to save
    shake_dict = {}
    total_loss = 0.0
    for key, value in loaded_dict.items():
        target = value['targets'].squeeze(-1)
        shake_result_spectrum = pd.read_csv(
            f"{shake_data_path}/{value['file_names'][1]}.csv", header=0).to_numpy()
        shake_result_spectrum = shake_result_spectrum.transpose()
        error_evolution = (shake_result_spectrum - target)**2
        loss = error_evolution.sum() / len(error_evolution.squeeze())  # MSE
        total_loss += loss
        shake_dict[key] = {
            'file_names': value['file_names'][1],
            'periods': value['periods'],
            'predictions': shake_result_spectrum,
            'error_evolution': error_evolution.squeeze(),
            'loss': loss
        }
    # Report total average loss
    avg_loss = total_loss / len(loaded_dict)
    # Save result
    save_avg_loss = {"avg_loss": avg_loss, "avg_time": None}
    with open(f"{shake_data_path}/avg_loss.json", "w") as out_file:
        json.dump(save_avg_loss, out_file, indent=4)

    # Get model results
    all_results = {}
    for model_id in model_ids:
        if model_id != 'shake':
            results_path = f'data/outputs/{site}-{model_id}/'
            # Load the results
            with open(f'{results_path}/results.pkl', 'rb') as file:
                loaded_dict = pickle.load(file)
            all_results[model_id] = loaded_dict
        else:
            all_results[model_id] = shake_dict

    # Reorder result data and append "shake" result
    result_dict = {}
    for site_id in site_ids:
        result_dict[site_id] = {}
        for model_id in model_ids:
            result_dict[site_id][model_id] = all_results[model_id][site_id]

    # Preprocess period
    periods_array = np.concatenate(periods)

    n_sites = 0
    running_sum_errors = {}
    for model_id in model_ids:
        running_sum_errors[model_id] = np.zeros(periods_array.shape)

    # Plot response spectrum per model
    for site_id, results in result_dict.items():
        print(f"Plot site {site_id}")
        target = results[model_ids[0]]["targets"][0].squeeze()

        # Init fig
        fig, ax = plt.subplots(figsize=(4, 2.8))

        # Target (Truth)
        ax.plot(
            periods_array, target,
            linestyle='-',
            color='black',
            alpha=0.8,
            label="True")

        # Predictions
        for model, result in results.items():
            ax.plot(
                periods_array, result["predictions"][0],
                linestyle=config[model]['linestyle'],
                color=config[model]['color'],
                label=model)

        # Fig settings
        ax.set_xlabel("Period (sec)")
        ax.set_ylabel("SA (g)")
        ax.set_xlim([0.01, 10])
        ax.set_ylim([0, None])
        ax.set_xscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{site_id}.{fig_format}')

        # Error
        n_sites += 1
        for model, result in results.items():
            current_error_evolution = result['error_evolution']
            running_sum_errors[model] += current_error_evolution

    # Avg error for this site
    avg_error = {}
    fig, ax = plt.subplots(figsize=(4, 2.8))
    for model_id in model_ids:
        model_error = np.sqrt(running_sum_errors[model_id] / n_sites)
        avg_error[model_id] = model_error

        # Plot avg error
        ax.plot(periods_array, model_error,
                linestyle=config[model_id]['linestyle'],
                color=config[model_id]['color'],
                label=model_id)
    ax.set_xlabel("Period (sec)")
    ax.set_ylabel("Avg. error (g)")
    ax.set_xlim([0.01, 10])
    ax.set_ylim(0, None)
    ax.set_xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{site}-avg_error.{fig_format}')

    with open(f'{save_dir}/{site}-avg_error.pkl', 'wb') as file:
        pickle.dump(avg_error, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{save_dir}/{site}-all_results.pkl', 'wb') as file:
        pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


# Global error
def global_error(sites):

    # Save dir
    save_dir = f'data/analysis/'
    periods_array = np.concatenate(periods)

    # Gather avg errors per site
    avg_errors = {}
    for site in sites:
        analysis_dir = f'data/analysis/{site}'

        with open(f'{analysis_dir}/{site}-avg_error.pkl', 'rb') as file:
            avg_error = pickle.load(file)

        avg_errors[site] = avg_error

    # Compute global avg error
    global_errors = {}
    models = list(config.keys())
    sequence_len = len(avg_error[models[0]])
    for model in models:
        global_errors[model] = np.zeros(sequence_len)

    for model in models:
        n_sites = 0
        for site, results in avg_errors.items():
            n_sites += 1
            global_errors[model] += results[model]
    for key, value in global_errors.items():
        global_errors[key] = value / n_sites

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for model in models:
        # Plot avg error
        ax.plot(periods_array, global_errors[model],
                linestyle=config[model]['linestyle'],
                color=config[model]['color'],
                label=model)
    ax.set_xlabel("Period (sec)")
    ax.set_ylabel("Global error (g)")
    ax.set_xlim([0.01, 10])
    ax.set_ylim(0, None)
    ax.set_xscale('log')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{save_dir}/global_error.{fig_format}')

    with open(f'{save_dir}/global_error.pkl', 'wb') as file:
        pickle.dump(global_errors, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_mse(sites):

    # Make dict to contain mse
    mse_errors = {}
    models = list(config.keys())
    for model in models:
        mse_errors[model] = []

    # Fill the dict with mse
    for model in models:
        for site in sites:
            # Load result
            result_path = f'data/analysis/{site}/{site}-all_results.pkl'
            with open(result_path, 'rb') as file:
                result = pickle.load(file)

            # Compute mse
            total_mse = 0.0
            n_data = len(result)
            for key, value in result.items():
                mse = np.mean(value[model]['error_evolution'])
                total_mse += mse
            avg_mse = total_mse / n_data
            mse_errors[model].append(avg_mse)

    df = pd.DataFrame.from_dict(mse_errors)


sites = ("FKSH17", "IWTH21", "FKSH18", "FKSH19", "IBRH13", "IWTH02", "IWTH05", "IWTH12", "IWTH14", "IWTH22", "IWTH27", "MYGH04")
# sites = ("FKSH17", "IWTH21", "FKSH18")

# Site analysis
for site_name in sites:
    site_analysis(site_name)

# Make global error
global_error(sites=sites)

# MSE
get_mse(sites)


a = 1