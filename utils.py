import numpy as np
import eqsig.single
import torch


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
    weights = torch.tensor([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=torch.float32).to(device)

    # Normalize the weights so that they sum to 1
    weights /= weights.sum()

    # The result tensor will have the same shape as the input but will be of type float due to division
    result = torch.full(sequence.shape, float('nan'))

    # Apply the weighted moving average to each element in the sequence
    for i in range(4, sequence.size(0) - 4):
        # The window includes the current element and 4 elements on each side
        window = sequence[i - 4:i + 5]

        # Calculate the weighted average for the window
        result[i] = torch.dot(window, weights)

    return result.unsqueeze(0)