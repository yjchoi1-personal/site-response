from functools import partial
import torch
import os
from torch import nn
import models
import utils
import data_loader
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle


site_id = 'FKSH17'
model_id = 'debug'
model_type = "lstm"  # "cnn" or "transformer" or "simpleCNN"
normalize_type = "standardization"  # "minmax" or "standardization"
mode = "train"  # "train" or "test"
train_batch = 4
valid_batch = 10
num_epochs = 100
resume = False

work_dir = '/work2/08264/baagee/frontera/site-response/'
data_path = f'{work_dir}/data/datasets/{site_id}/'
train_data_path = f'{data_path}/spectrum_train.npz'
test_data_path = f'{data_path}/spectrum_test.npz'
output_path = f'{work_dir}/data/outputs/{site_id}-{model_id}/'
checkpoint_path = f'{work_dir}/data/checkpoints/{site_id}-{model_id}/'
checkpoint_file = 'checkpoint.pth'
period_ranges = ((0.01, 0.1, 167), (0.1, 1, 167), (1, 10, 166))
valid = True

config = {
    'embedding_size': tune.choice([4, 8, 16, 32, 64]),
    'nhead': tune.choice([1, 2, 4, 8]),
    'num_encoder_layers': tune.choice([1, 2, 3, 6]),
    'num_decoder_layers': tune.choice([1, 2, 3, 6]),
    'dim_feedforward': tune.choice([64, 128, 256, 512, 1024, 2048]),
    'dropout': tune.choice([0.0, 0.1, 0.2]),
    "positional_encoding": tune.choice([False, True])
}





def validate_model(
        model,
        ds_valid,
        normalize_stats, normalize_type,
        device):

    model.eval()
    criterion = nn.MSELoss()
    valid_running_loss = 0.0

    with torch.no_grad():
        for _, (inputs, targets) in ds_valid:
            inputs, targets = inputs.to(device), targets.to(device)
            normalized_inputs = utils.normalize_inputs(inputs, normalize_stats, normalize_type)
            outputs = model(normalized_inputs)
            loss = criterion(outputs, targets.squeeze())
            valid_running_loss += loss.item()

    return valid_running_loss / len(ds_valid)


def train_cifar(config, train_data_path, test_data_path, device):

    # Load training and validation data
    ds_train, train_statistics = data_loader.get_data(
        path=train_data_path, batch_size=train_batch)
    ds_valid, _ = data_loader.get_data(
        path=test_data_path, batch_size=valid_batch, shuffle=True)

    # Get necessary variables about data from dataset
    ds_iterator = iter(ds_train)
    ds_batch = next(ds_iterator)
    sequence_length = ds_batch[1][0].shape[1]
    n_features = ds_batch[1][0].shape[-1]

    # Set up the model
    model = utils.init_model(
        model_type, sequence_length, n_features,
        embedding_size=config['embedding_size'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        positional_encoding=config['positional_encoding'])

    model = nn.DataParallel(model)
    model.to(device)

    # Normalization stat
    normalize_stats = {
        "mean": torch.tensor(train_statistics["feature_mean"]).to(torch.float32).to(device),
        "std": torch.tensor(train_statistics["feature_std"]).to(torch.float32).to(device),
        "max": torch.tensor(train_statistics["feature_max"]).to(torch.float32).to(device),
        "min": torch.tensor(train_statistics["feature_min"]).to(torch.float32).to(device)
    }

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize empty lists to store loss histories
    train_losses = []
    valid_losses = []

    start_epoch = 0
    iteration = 0

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_state = models.load_checkpoint(f'{checkpoint_path}/{checkpoint_file}')
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            start_epoch = checkpoint_state["epoch"] + 1
            iteration = checkpoint_state['iteration']
            print(f"Resume start from {start_epoch}")
    else:
        start_epoch = 0

    # Training loop
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

        model.eval()
        criterion = nn.MSELoss()
        valid_running_loss = 0.0

        with torch.no_grad():
            for _, (inputs, targets) in ds_valid:
                inputs, targets = inputs.to(device), targets.to(device)
                normalized_inputs = utils.normalize_inputs(inputs, normalize_stats, normalize_type)
                outputs = model(normalized_inputs)
                loss = criterion(outputs, targets.squeeze())
                valid_running_loss += loss.item()

        valid_loss = valid_running_loss / len(ds_valid)

        models.save_checkpoint({
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, filename=f'{checkpoint_path}/{checkpoint_file}')

        checkpoint = Checkpoint.from_directory(f'{checkpoint_path}')
        train.report(
            {"loss": valid_loss},
            checkpoint=checkpoint,
        )


    with open(f'{checkpoint_path}/loss_histories.pkl', 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'valid_losses': valid_losses}, f)


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

    return avg_loss


def main(num_samples=4, max_num_epochs=100, gpus_per_trial=1):

    sequence_length = 500
    n_features = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and validation data
    _, train_statistics = data_loader.get_data(
        path=train_data_path, batch_size=train_batch)

    # Normalization stat
    normalize_stats = {
        "mean": torch.tensor(train_statistics["feature_mean"]).to(torch.float32).to(device),
        "std": torch.tensor(train_statistics["feature_std"]).to(torch.float32).to(device),
        "max": torch.tensor(train_statistics["feature_max"]).to(torch.float32).to(device),
        "min": torch.tensor(train_statistics["feature_min"]).to(torch.float32).to(device)
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(
            train_cifar,
            train_data_path=train_data_path, test_data_path=test_data_path, device=device),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    best_trained_model = utils.init_model(
        model_type, sequence_length, n_features,
        embedding_size=best_trial.config['embedding_size'],
        nhead=best_trial.config['nhead'],
        num_encoder_layers=best_trial.config['num_encoder_layers'],
        num_decoder_layers=best_trial.config['num_decoder_layers'],
        dim_feedforward=best_trial.config['dim_feedforward'],
        dropout=best_trial.config['dropout'],
        positional_encoding=best_trial.config['positional_encoding'])

    best_checkpoint_data = result.get_best_checkpoint(
        trial=best_trial, metric="loss", mode="min")
    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    ds_test, _ = data_loader.get_data(
        path=test_data_path, batch_size=1, shuffle=True)

    test_loss = predict(
        model=best_trained_model,
        ds_test=ds_test,
        normalize_stats=normalize_stats,
        normalize_type=normalize_type,
        checkpoint_path=checkpoint_path,
        checkpoint_file=checkpoint_file,
        output_path=output_path,
        device=device
    )
    # print("Best trial test set accuracy: {}".format(test_loss))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)