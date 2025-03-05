import os
import sys
import torch
import pytest

# Add the parent directory to the path so we can import from the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models
import utils
import data_loader

def test_training_runs_for_limited_iterations():
    """Test that training runs for a limited number of iterations"""
    # Setup test parameters
    site = "FKSH19"
    model_type = "cnn"
    max_iterations = 5
    
    # Set up paths
    data_path = f'data/datasets/{site}/'
    train_data_path = f'{data_path}/spectrum_train.npz'
    test_data_path = f'{data_path}/spectrum_test.npz'
    
    # Load config
    with open("config.json") as config_file:
        import json
        config = json.load(config_file)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_batch = 4
    ds_train, train_statistics = data_loader.get_data(
        path=train_data_path, batch_size=train_batch)
    
    # Normalization stats
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
    
    # Initialize model
    hyperparams = config['model'].get(model_type)
    model = utils.init_model(
        model_type, sequence_length, n_features,
        **hyperparams
    ).to(device)
    
    # Create a custom training function that stops after max_iterations
    class EarlyStopException(Exception):
        pass
    
    # Track iterations
    iteration_count = 0
    
    # Load training data
    ds_train, _ = data_loader.get_data(
        path=train_data_path, batch_size=train_batch)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'])
    criterion = torch.nn.MSELoss()
    
    try:
        for epoch in range(config['optimizer']['num_epochs']):
            model.train()  # Set the model to training mode
            
            for _, (inputs, targets) in ds_train:
                inputs, targets = inputs.to(device), targets.to(device)
                normalized_inputs = utils.normalize_inputs(
                    inputs, normalize_stats, "standardization")
                outputs = model(normalized_inputs)
                loss = criterion(outputs, targets.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                iteration_count += 1
                if iteration_count >= max_iterations:
                    raise EarlyStopException("Reached maximum iterations")
    
    except EarlyStopException:
        pass
    
    # Assert that we ran for the expected number of iterations
    assert iteration_count == max_iterations, f"Expected {max_iterations} iterations, but ran {iteration_count}"
    
    # Print success message
    print("Training test completed successfully!")

if __name__ == "__main__":
    test_training_runs_for_limited_iterations() 