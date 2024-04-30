import os.path
import matplotlib.pyplot as plt
import torch
import numpy as np
import data_loader
import model_ann
import torch.nn as nn
import pickle
import utils


site_id = 'FKSH17'
data_path = f'data/datasets/{site_id}/'
train_data_path = f'{data_path}/spectrum_train.npz'
test_data_path = f'{data_path }/spectrum_test.npz'
output_path = f'data/outputs/{site_id}/'
checkpoint_path = f'data/checkpoints/{site_id}/'
checkpoint_file = 'checkpoint.pth'
resume = False
num_epochs = 100
mlp_hidden_dim = 128
mode = "test"  # "train" or "test"
period_ranges = ((0.01, 0.1, 167), (0.1, 1, 167), (1, 10, 166))
valid = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize empty lists to store loss histories
train_losses = []
valid_losses = []

# set folders for model checkpoint and testoutputs
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)

# load training ata
ds_train, train_statistics = data_loader.get_data(
    path=train_data_path, batch_size=10)

# Normalization stat
mean = torch.tensor(train_statistics["feature_mean"]).to(torch.float32).to(device)
std = torch.tensor(train_statistics["feature_std"]).to(torch.float32).to(device)

if mode == 'train':
    if valid:
        ds_valid, valid_statistics = data_loader.get_data(
            path=test_data_path, batch_size=4, shuffle=False)

    # Get necessary variables about data from dataset
    ds_iterator = iter(ds_train)
    ds_batch = next(ds_iterator)
    sequence_length = ds_batch[0].shape[1]
    n_features = ds_batch[0].shape[-1]

    # init model
    model = model_ann.SequenceLSTM(sequence_length, n_features)
    # init loss measure
    criterion = nn.MSELoss()
    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Try to load existing checkpoint
    if resume:
        checkpoint = model_ann.load_checkpoint(f'{checkpoint_path}/{checkpoint_file}')
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resume start from {start_epoch}")
    else:
        start_epoch = 0

    # Move the model to the appropriate device
    model.to(device)

    # Start training
    iteration = 0 if not resume else checkpoint['iteration']

    for epoch in np.arange(start_epoch, num_epochs):
        model.train()  # Set the model to training mode
        train_running_loss = 0.0

        for _, (inputs, targets) in ds_train:
            # Move data to the same device as the model
            # inputs=(None, 500, 3), targets=(None, 500, 1)
            inputs, targets = inputs.to(device), targets.to(device)

            # Normalize inputs

            normalized_inputs = ((inputs - mean) / std)

            # Forward pass: compute the model output
            outputs = model(normalized_inputs)

            # Compute the loss
            loss = criterion(outputs, targets.squeeze())

            # Backward pass and optimize
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update model parameters

            # Track the loss
            print(f"Step {iteration}: {loss.item():.4f}")
            train_losses.append([iteration, loss.item()])
            train_running_loss += loss.item()
            iteration += 1

        # Print average loss for the epoch
        train_loss = train_running_loss / len(ds_train)

        # Valid
        if valid:
            # Validation phase
            model.eval()
            valid_running_loss = 0.0
            with torch.no_grad():
                for _, (inputs, targets) in ds_valid:
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Normalize inputs
                    normalized_inputs = ((inputs - mean) / std)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets.squeeze())
                    valid_losses.append([iteration, loss.item()])
                    valid_running_loss += loss.item()
            valid_loss = valid_running_loss / len(ds_valid)


        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        if valid:
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {valid_loss:.4f}")

        # Save the model checkpoint
        model_ann.save_checkpoint({
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, filename=f'{checkpoint_path}/{checkpoint_file}')

    with open(f'{checkpoint_path}/loss_histories.pkl', 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'valid_losses': valid_losses}, f)

elif mode == "test":
    ds, _ = data_loader.get_data(
        path=test_data_path, batch_size=1, shuffle=False)

    # define iterator for use in training
    ds_iterator = iter(ds)
    # extract batch
    ds_batch = next(ds_iterator)
    sequence_length = ds_batch[1][0].shape[1]
    n_features = ds_batch[1][0].shape[-1]

    model = model_ann.SequenceLSTM(
        sequence_length, n_features, mlp_hidden_dim=128, nmlp_layers=2)
    checkpoint = torch.load(f'{checkpoint_path}/{checkpoint_file}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():  # No need to track gradients during evaluation
        for i, (file_names, (inputs, targets)) in enumerate(ds):
            # Move data to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)

            # Normalize inputs
            normalized_inputs = ((inputs - mean) / std)

            # Forward pass: compute the model output
            outputs = model(normalized_inputs)
            # Apply weighted moving average to remove prediction noise
            outputs_smooth = utils.smoothen(outputs.squeeze(), device)

            # Compute the loss
            loss = criterion(outputs, targets.squeeze(-1))

            total_loss += loss.item()  # Sum up batch loss

            # Visualize prediction
            response_pred = outputs_smooth.cpu().numpy().squeeze(0)
            response_true = targets.cpu().numpy().squeeze(0)
            periods = [np.linspace(start, end, num, endpoint=False) for start, end, num in period_ranges]
            periods = np.concatenate(periods)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(periods, response_pred, linewidth=3, label="LSTM")
            ax.plot(periods, response_true, linewidth=3, label="True")
            ax.set_xlabel("Period (sec)")
            ax.set_ylabel("SA (g)")
            ax.set_xlim([0.01, 10])
            ax.set_xscale('log')
            ax.set_title(f"MSE={loss:.3e}")
            plt.tight_layout()
            plt.legend()
            plt.show()

    avg_loss = total_loss / len(ds)
    print(f'Average Test Loss: {avg_loss:.4f}')