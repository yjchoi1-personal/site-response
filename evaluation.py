import os
import torch
from torch import nn
import utils
import json
import pickle
import data_loader
import numpy as np
import pandas as pd
import time

# site = 'FKSH17'
# data_path = f'data/datasets/{site}/'
# test_data_path = f'{data_path}/spectrum_test.npz'


def predict(
        model,
        test_data_path,
        periods,
        normalize_stats,
        normalize_type,
        checkpoint_path,
        checkpoint_file,
        output_path,
        device):

    # Test data loader
    ds_test, _ = data_loader.get_data(
        path=test_data_path, batch_size=1, shuffle=False)

    # Set folders for test outputs
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"{checkpoint_path} not exist in {checkpoint_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Result container
    results = {}

    # Set the model to evaluation mode
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    # Load the model checkpoint
    checkpoint = torch.load(f'{checkpoint_path}/{checkpoint_file}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    begin = time.time()
    with torch.no_grad():  # No need to track gradients during evaluation
        for i, (file_names, (inputs, targets)) in enumerate(ds_test):
            inputs, targets = inputs.to(device), targets.to(device)
            normalized_inputs = utils.normalize_inputs(
                inputs, normalize_stats, normalize_type)

            # Forward pass: compute the model output
            outputs = model(normalized_inputs)
            # Apply weighted moving average to remove prediction noise
            outputs_smooth = utils.smoothen(outputs.squeeze(), device)

            # Compute the loss
            nan_range = [4, -4]  # Adjusted for smoothen function that removes first 4 and last 4 values
            loss = criterion(
                outputs_smooth[:, nan_range[0]:nan_range[1]],
                targets.squeeze(-1)[:, nan_range[0]:nan_range[1]])
            total_loss += loss.item()

            # Fill NaN to the adjacent values when saving
            outputs_smooth_np = outputs_smooth.cpu().numpy()[0]
            s = pd.Series(outputs_smooth_np)
            s = s.ffill()
            s = s.bfill()
            outputs_smooth_processed = s.values
            outputs_smooth_processed = outputs_smooth_processed.reshape(1, -1)

            # Compute error_evolution
            targets_np = targets.cpu().numpy().squeeze(-1)
            error_evolution = (targets_np - outputs_smooth_processed)**2  # (1, 500, 1)
            error_evolution = error_evolution.squeeze(0)  # (1, 500)

            # Save results
            site = file_names[0][0].split('_')[0]
            results[f"{site}-{i}"] = {
                "file_names": [file_names[0][0], file_names[1][0]],
                "periods": periods,
                "inputs": inputs.cpu().numpy(),
                "targets": targets.cpu().numpy(),
                "predictions": outputs_smooth_processed,
                "error_evolution": error_evolution,
                "loss": loss.item()
            }

            # Visualize prediction
            response_pred = outputs_smooth.cpu().numpy().squeeze(0)
            response_true = targets.cpu().numpy().squeeze(0)
            utils.test_vis(
                periods,
                response_pred, response_true, loss,
                output_path, file_names, i)

    end = time.time()
    total_time = end - begin
    # Report total average loss
    avg_loss = total_loss / len(ds_test)
    print(f'Average Test Loss: {avg_loss:.3e}')
    # Report avg time for single prediction
    avg_time = total_time / len(ds_test)

    # Save result
    save_avg_loss = {"avg_loss": avg_loss, "avg_time": avg_time}
    with open(f"{output_path}/avg_loss.json", "w") as out_file:
        json.dump(save_avg_loss, out_file, indent=4)

    with open(f"{output_path}/results.pkl", 'wb') as file:
        pickle.dump(results, file)
