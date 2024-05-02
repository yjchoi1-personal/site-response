import os.path
import matplotlib.pyplot as plt
import torch
import numpy as np
import data_loader
import models
import torch.nn as nn
import pickle
import utils
import argparse
import json


site_id = 'FKSH17'
model_id = 'debug'
model_type = "lstm2"  # "cnn" or "transformer" or "simpleCNN"
normalize_type = "standardization"  # "minmax" or "standardization"
mode = "train"  # "train" or "test"
train_batch = 4
valid_batch = 10
num_epochs = 100
resume = False

# For transformer
positional_encoding = False

# For lstm2
n_lstm_layers = 3
hidden_dim = 32

data_path = f'data/datasets/{site_id}/'
train_data_path = f'{data_path}/spectrum_train.npz'
test_data_path = f'{data_path}/spectrum_test.npz'
output_path = f'data/outputs/{site_id}-{model_id}/'
checkpoint_path = f'data/checkpoints/{site_id}-{model_id}/'
checkpoint_file = 'checkpoint.pth'
period_ranges = ((0.01, 0.1, 167), (0.1, 1, 167), (1, 10, 166))
valid = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()


def train(
        model, ds_train, ds_valid,
        normalize_stats, normalize_type,
        num_epochs,
        checkpoint_path,
        checkpoint_file,
        valid,
        device):

    # set folders for model checkpoint.
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize empty lists to store loss histories
    train_losses = []
    valid_losses = []

    start_epoch = 0
    iteration = 0

    # Try to load existing checkpoint
    if resume:
        checkpoint = models.load_checkpoint(f'{checkpoint_path}/{checkpoint_file}')
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            iteration = checkpoint['iteration']
            print(f"Resume start from {start_epoch}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set the model to training mode
        train_running_loss = 0.0

        for _, (inputs, targets) in ds_train:
            inputs, targets = inputs.to(device), targets.to(device)
            normalized_inputs = utils.normalize_inputs(
                inputs, normalize_stats, normalize_type)
            outputs = model(normalized_inputs)
            loss = criterion(outputs, targets.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the loss
            print(f"Step {iteration}: {loss.item():.4e}")
            train_losses.append([iteration, loss.item()])
            train_running_loss += loss.item()
            iteration += 1

        train_loss = train_running_loss / len(ds_train)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4e}")

        if valid:
            valid_loss = validate_model(
                model, ds_valid, normalize_stats, normalize_type, device)
            valid_losses.append([iteration, valid_loss])
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {valid_loss:.4e}")

        models.save_checkpoint({
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, filename=f'{checkpoint_path}/{checkpoint_file}')

    with open(f'{checkpoint_path}/loss_histories.pkl', 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'valid_losses': valid_losses}, f)


def validate_model(
        model,
        ds_valid,
        normalize_stats, normalize_type,
        device):

    model.eval()
    valid_running_loss = 0.0

    with torch.no_grad():
        for _, (inputs, targets) in ds_valid:
            inputs, targets = inputs.to(device), targets.to(device)
            normalized_inputs = utils.normalize_inputs(inputs, normalize_stats, normalize_type)
            outputs = model(normalized_inputs)
            loss = criterion(outputs, targets.squeeze())
            valid_running_loss += loss.item()

    return valid_running_loss / len(ds_valid)


def predict(
        model,
        ds_test,
        normalize_stats, normalize_type,
        checkpoint_path, checkpoint_file,
        output_path,
        device):

    # Set folders for test outputs
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"{checkpoint_path} not exist in {checkpoint_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Set the model to evaluation mode
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    # Load the model checkpoint
    checkpoint = torch.load(f'{checkpoint_path}/{checkpoint_file}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

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

            # Visualize prediction
            response_pred = outputs_smooth.cpu().numpy().squeeze(0)
            response_true = targets.cpu().numpy().squeeze(0)
            utils.test_vis(
                period_ranges,
                response_pred, response_true, loss,
                output_path, file_names, i)

    # Report total average loss
    avg_loss = total_loss / len(ds_test)
    print(f'Average Test Loss: {avg_loss:.3e}')

    # Save loss result
    save_avg_loss = {"avg_loss": avg_loss}
    with open(f"{output_path}/avg_loss.json", "w") as out_file:
        json.dump(save_avg_loss, out_file, indent=4)


if __name__ == '__main__':

    # Load training and validation data
    ds_train, train_statistics = data_loader.get_data(
        path=train_data_path, batch_size=train_batch)
    ds_valid, _ = data_loader.get_data(
        path=test_data_path, batch_size=valid_batch, shuffle=True)

    # Normalization stat
    normalize_stats = {
        "mean": torch.tensor(train_statistics["feature_mean"]).to(torch.float32).to(device),
        "std": torch.tensor(train_statistics["feature_std"]).to(torch.float32).to(device),
        "max": torch.tensor(train_statistics["feature_max"]).to(torch.float32).to(device),
        "min": torch.tensor(train_statistics["feature_min"]).to(torch.float32).to(device)
    }

    # Get necessary variables about data from dataset
    ds_iterator = iter(ds_train)
    ds_batch = next(ds_iterator)
    sequence_length = ds_batch[1][0].shape[1]
    n_features = ds_batch[1][0].shape[-1]

    # Initiate model
    model = utils.init_model(
        model_type, sequence_length, n_features,
        positional_encoding=positional_encoding,
        n_lstm_layers=n_lstm_layers, hidden_dim=hidden_dim
    ).to(device)

    if mode == "train":
        train(
            model=model,
            ds_train=ds_train,
            ds_valid=ds_valid,
            normalize_stats=normalize_stats,
            normalize_type=normalize_type,
            num_epochs=num_epochs,
            checkpoint_path=checkpoint_path,
            checkpoint_file=checkpoint_file,
            valid=True,
            device=device
        )

    elif mode == "test":

        ds_test, _ = data_loader.get_data(
            path=test_data_path, batch_size=1, shuffle=True)

        predict(
            model=model,
            ds_test=ds_test,
            normalize_stats=normalize_stats,
            normalize_type=normalize_type,
            checkpoint_path=checkpoint_path,
            checkpoint_file=checkpoint_file,
            output_path=output_path,
            device=device
        )