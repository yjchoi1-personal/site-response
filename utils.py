import numpy as np
import eqsig.single
import torch
import models
import pickle
import matplotlib.pyplot as plt
import numpy as np


def compare_models():
    pass


def plot_loss(data_path, save_path):
    with open(data_path, 'rb') as f:
        loss_data = pickle.load(f)
        train_losses = loss_data['train_losses']
        valid_losses = loss_data['valid_losses']

    # Assuming 'train_losses' and 'valid_losses' are loaded
    plt.figure(figsize=(5, 3.5))
    plt.plot(
        [loss[0] for loss in train_losses], [loss[1] for loss in train_losses], label='Train')
    if valid_losses:
        plt.plot([loss[0] for loss in valid_losses], [loss[1] for loss in valid_losses], label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def test_vis(
        periods,
        response_pred,
        response_true,
        loss,
        output_path,
        file_names,
        index):
    """
    Plots the predicted and true response spectra.

    Args:
    period_ranges (tuple): Tuple of period ranges for the spectrum.
    outputs_smooth (torch.Tensor): Smoothed model outputs.
    targets (torch.Tensor): True target values.
    file_names (list): List of file names associated with the data.
    loss (float): Computed loss for the batch.
    output_path (str): Path where the plot should be saved.
    index (int): Index of the plot for file naming.

    Returns:
    None
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(periods, response_pred,
            linewidth=3, color='black', label="Pred")
    ax.plot(periods, response_true,
            linewidth=3, color='silver', label="True")
    ax.set_xlabel("Period (sec)")
    ax.set_ylabel("SA (g)")
    ax.set_xlim([0.01, 10])
    ax.set_ylim([0, None])
    ax.set_xscale('log')
    ax.set_title(f"x={file_names[0][0]}, y={file_names[1][0]}, MSE={loss:.3e}", fontsize=10)
    plt.tight_layout()
    plt.legend()

    # Save the plot
    plt.savefig(f"{output_path}/site{index}-{file_names[0]}.png")
    plt.close(fig)


def init_model(
        model_type, sequence_length, n_features,
        **kwargs):
    """
    Initiate model
    Args:
        model_type (str): "lstm", "cnn", "transformer", "simpleCNN"
        sequence_length (int): array length of spectrum values
        n_features (int): number of input features
        positional_encoding (bool): only for transformer
        **kwargs: additional keyword arguments specific to different models

    Returns:
        model object
    """

    # init model
    if model_type == "lstm":
        relevant_kwargs = {
            key: kwargs[key] for key in [
                'n_lstm_layers',
                "hidden_dim"
            ] if key in kwargs
        }
        model = models.SequenceLSTM(
            sequence_length, n_features, **relevant_kwargs)

    elif model_type == "lstm2":
        relevant_kwargs = {
            key: kwargs[key] for key in [
                'n_lstm_layers',
                "hidden_dim"
            ] if key in kwargs
        }
        model = models.SequenceLSTM2(
            sequence_length, n_features, **relevant_kwargs)

    elif model_type == "cnn":
        relevant_kwargs = {
            key: kwargs[key] for key in [
                'out_channels',
                'kernel_sizes',
                'pool_sizes'
            ] if key in kwargs
        }
        model = models.Conv1D(
            sequence_length, n_features, **relevant_kwargs)

    elif model_type == "transformer":
        relevant_kwargs = {
            key: kwargs[key] for key in [
                'embedding_size',
                'nhead',
                'num_encoder_layers',
                'num_decoder_layers',
                'dim_feedforward',
                'dropout',
                'positional_encoding'
            ] if key in kwargs
        }
        model = models.TimeSeriesTransformer(
            sequence_length, n_features,
            **relevant_kwargs)

    elif model_type == "transformer2":
        relevant_kwargs = {
            key: kwargs[key] for key in [
                'positional_encoding'
            ] if key in kwargs
        }
        model = models.TimeSeriesTransformer2(
            sequence_length, n_features,
            **relevant_kwargs
        )

    elif model_type == "simpleCNN":
        model = models.simpleCNN(
            sequence_length, n_features)
    else:
        raise ValueError

    return model


def normalize_inputs(inputs, normalize_stats, option):
    """
    Normalize inputs: "minmax" or "standardization"
    Args:
        inputs (torch.tensor): inputs with shape=(sequence_length, n_features)
        normalize_stats (dict): a dictionary that contains statistics of the input features
        option (str): "standardization" or "minmax"
    Returns:
    Normalized input
    """

    if option == "standardization":
        normalized_inputs = (
                (inputs - normalize_stats["mean"]) / normalize_stats["std"])
    elif option == "minmax":
        normalized_inputs = (inputs - normalize_stats["min"]) / (normalize_stats["max"] - normalize_stats["min"])
    else:
        raise ValueError

    return normalized_inputs


def time2freq(timeseries_data, dt, periods):
    sa_period_results = []
    for period in periods:
        record = eqsig.AccSignal(timeseries_data, dt)
        record.generate_response_spectrum(response_times=period)
        sa_period_results.append(record.s_a)
    frequency_data = np.hstack(sa_period_results)
    return frequency_data


def smoothen(sequence, device):
    """
    Applies a weighted moving average filter to a sequence of data using PyTorch.
    Args:
    - sequence (Tensor): The input sequence of data points with shape (n_sequence,).
    Returns:
    - Tensor: The smoothed sequence with the weighted moving average applied, same shape as input.
    """
    # change the shape


    # Define the weights for the moving average
    weights = torch.tensor(
        [1, 2, 3, 4, 5, 4, 3, 2, 1],
        dtype=torch.float32).to(device)

    # Normalize the weights so that they sum to 1
    weights /= weights.sum()

    # The result tensor will have the same shape as the input but will be of type float due to division
    result = torch.full(sequence.shape, float('nan')).to(device)

    # Apply the weighted moving average to each element in the sequence
    for i in range(4, sequence.size(0) - 4):
        # The window includes the current element and 4 elements on each side
        window = sequence[i - 4:i + 5]

        # Calculate the weighted average for the window
        result[i] = torch.dot(window, weights)

    return result.unsqueeze(0)