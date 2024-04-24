import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import os

def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:
    """Build a MultiLayer Perceptron.

    Args:
      input_size: Size of input layer.
      layer_sizes: An array of input size for each hidden layer.
      output_size: Size of the output layer.
      output_activation: Activation function for the output layer.
      activation: Activation function for the hidden layers.

    Returns:
      mlp: An MLP sequential container.
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [activation for i in range(nlayers)]
    act[-1] = output_activation

    # Create a torch sequential container
    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                                 layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())

    return mlp

class Conv1D(nn.Module):
    def __init__(self, sequence_length, n_features, mlp_hidden_dim=None, nmlp_layers=None):
        super(Conv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=16, kernel_size=24)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=12)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=6)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Dummy input to calculate the shape after conv and pool layers
        dummy_input = torch.randn(1, n_features, sequence_length)
        dummy_output = self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))))
        self.flat_features = dummy_output.numel()

        # Dense layer
        self.dense = nn.Linear(self.flat_features, sequence_length)

        # # MLP layer
        # self.mlp = build_mlp(
        #     self.flat_features, [mlp_hidden_dim for _ in range(nmlp_layers)], output_size=sequence_length)
        # self.layer_norm = nn.LayerNorm(sequence_length)

    def forward(self, x):
        # Reshape input to (batch_size, n_features, sequence_length)
        x = x.permute(0, 2, 1)
        # Shape: (batch_size, n_features, time_steps) -> (batch_size, 16, time_steps - 24 + 1)
        x = self.conv1(x)
        x = F.relu(x)
        # Shape: (batch_size, 16, time_steps - 24 + 1) -> (batch_size, 16, (time_steps - 24 + 1) / 2)
        x = self.pool1(x)
        # Shape: (batch_size, 16, (time_steps - 24 + 1) / 2) -> (batch_size, 32, ((time_steps - 24 + 1) / 2 - 12 + 1))
        x = self.conv2(x)
        x = F.relu(x)
        # Shape: (batch_size, 32, ((time_steps - 24 + 1) / 2 - 12 + 1)) -> (batch_size, 32, (((time_steps - 24 + 1) / 2 - 12 + 1) / 2))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense(x)

        # x = self.mlp(x)

        return x

class SequenceLSTM(nn.Module):
    def __init__(self, sequence_length, n_features, mlp_hidden_dim=None, nmlp_layers=None):
        super(SequenceLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=32, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Use a theoretical output size for flat features calculation
        # Assuming output of second LSTM is (batch_size, sequence_length, 32)
        self.flat_features = sequence_length * 64
        self.dense = nn.Linear(self.flat_features, sequence_length)

        # # MLP
        # self.mlp = build_mlp(
        #     self.flat_features, [mlp_hidden_dim for _ in range(nmlp_layers)], output_size=sequence_length)
        # self.layer_norm = nn.LayerNorm(sequence_length)

    def forward(self, x):

        # Process through LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        # Flatten the output
        x = self.flatten(x)

        # Dense
        x = self.dense(x)

        # # Pass through MLP and normalize
        # x = self.mlp(x)
        # x = self.layer_norm(x)

        return x

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def load_checkpoint(filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        print("Loaded checkpoint '{}'".format(filename))
        return checkpoint
    else:
        print("No checkpoint found at '{}'".format(filename))
        return None
