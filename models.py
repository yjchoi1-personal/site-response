import torch
import math
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


class simpleCNN(nn.Module):
    def __init__(self, sequence_length, n_features):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(n_features, sequence_length, kernel_size=(21, 1))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Dummy input to calculate the shape after conv and pool layers
        dummy_input = torch.randn(1, n_features, sequence_length, 1)
        dummy_output = self.flatten(self.relu(self.conv1(dummy_input)))
        self.flat_features = dummy_output.numel()

        self.fc1 = nn.Linear(self.flat_features, sequence_length)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softplus(x)
        return x


class Conv1D(nn.Module):
    def __init__(self, sequence_length, n_features, mlp_hidden_dim=None, nmlp_layers=None):
        super(Conv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=16, kernel_size=24)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=12)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=6)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()

        # Dummy input to calculate the shape after conv and pool layers
        dummy_input = torch.randn(1, n_features, sequence_length)
        dummy_output = self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))))
        self.flat_features = dummy_output.numel()

        # Dense layer
        self.dense = nn.Linear(self.flat_features, sequence_length)
        self.softplus = nn.Softplus()

        # # MLP layer
        # self.mlp = build_mlp(
        #     self.flat_features, [mlp_hidden_dim for _ in range(nmlp_layers)], output_size=sequence_length)
        # self.layer_norm = nn.LayerNorm(sequence_length)

    def forward(self, x):
        # Reshape input to (batch_size, embedding, sequence_length)
        x = x.permute(0, 2, 1)

        # Shape: (batch_size, embedding, time_steps) -> (batch_size, 16, time_steps - 24 + 1)
        x = self.conv1(x)
        x = self.relu1(x)

        # Shape: (batch_size, 16, time_steps - 24 + 1) -> (batch_size, 16, (time_steps - 24 + 1) / 2)
        x = self.pool1(x)

        # Shape: (batch_size, 16, (time_steps - 24 + 1) / 2) -> (batch_size, 32, ((time_steps - 24 + 1) / 2 - 12 + 1))
        x = self.conv2(x)
        x = self.relu2(x)

        # Shape: (batch_size, 32, ((time_steps - 24 + 1) / 2 - 12 + 1)) -> (batch_size, 32, (((time_steps - 24 + 1) / 2 - 12 + 1) / 2))
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.softplus(x)

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


class SequenceLSTM2(nn.Module):
    def __init__(
            self, sequence_length, n_features, n_lstm_layers,
            hidden_dim, output_features=1):
        super(SequenceLSTM2, self).__init__()
        self.input_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = n_lstm_layers
        self.output_features = output_features

        # LSTM layer
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_dim,
                            num_layers=n_lstm_layers,
                            bidirectional=True,
                            batch_first=True)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Use a theoretical output size for flat features calculation
        # Assuming output of second LSTM is (batch_size, sequence_length, 32)
        self.flat_features = sequence_length * 64
        self.dense = nn.Linear(self.flat_features, sequence_length)

        # # Fully connected layer
        # self.fc = nn.Linear(hidden_dim, output_features)
        # # MLP
        # mlp_hidden_dim = 128
        # nmlp_layers = 2
        # self.decoder = nn.Sequential(
        #     *[build_mlp(
        #         hidden_dim*2, [mlp_hidden_dim for _ in range(nmlp_layers)], output_features)])

    def forward(self, x):
        # Forward propagate LSTM
        x, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_length, hidden_dim)

        # Flatten the output
        x = self.flatten(x)

        # Dense
        x = self.dense(x)

        # # Fully connected layer
        # x = self.decoder(x)
        # x = x.squeeze(-1)

        return x


class TimeSeriesTransformer2(nn.Module):
    def __init__(
            self,
            sequence_length,
            n_input_features,
            embedding_size=16,
            nhead=4,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0):

        super(TimeSeriesTransformer2, self).__init__()
        self.sequence_length = sequence_length
        self.n_input_features = n_input_features
        self.embedding_size = embedding_size

        # Embeddings for features and periods
        self.feature_embedding = nn.Linear(n_input_features, embedding_size)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_size, nhead, dim_feedforward, dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers)

        # Output layer
        self.output_layer = nn.Linear(embedding_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # Embed features and periods
        x = self.feature_embedding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Apply the final output layer
        x = self.output_layer(x)
        x = self.softplus(x)

        return x.squeeze(-1)


class TimeSeriesTransformer(nn.Module):
    def __init__(
            self,
            sequence_length,
            n_input_features,
            embedding_size=16,
            nhead=2,
            num_encoder_layers=3,
            dim_feedforward=2048,
            dropout=0.1,
            positional_encoding=False):
        super(TimeSeriesTransformer, self).__init__()

        # Model Hyperparameters
        self.sequence_length = sequence_length
        self.n_input_features = n_input_features
        self.embedding_size = embedding_size
        self.positional_encoding = positional_encoding

        # Embed
        self.dense = nn.Linear(n_input_features, embedding_size)

        # Positional encoding for adding notion of time step
        if self.positional_encoding:
            self.positional_encoder = PositionalEncoding(
                sequence_length, embedding_size)

        # Transformer Layer
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)

        # Output linear layer to match output dimensions
        self.output_linear = nn.Linear(embedding_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x shape is expected to be (nbatch, sequence_len, ndim)
        nbatch, sequence_len, ndim = x.shape

        # Reshape x to (-1, ndim) to apply the linear transformation
        x = x.reshape(-1, ndim)

        # x to latent dim
        x = self.dense(x)
        x = x.view(nbatch, sequence_len, self.embedding_size)

        # Add positional encoding
        if self.positional_encoding:
            x = self.positional_encoder(x)

        # Transformer
        x = self.transformer(x, x)  # Encoder self-attention

        # Pass through the output linear layer
        x = self.output_linear(x)
        x = self.softplus(x)

        return x.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, embedding_size):
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(sequence_length, embedding_size)

    def forward(self, x):
        device = x.device
        positions = torch.arange(0, self.sequence_length, device=device)
        embedded_positions = self.embedding(positions)
        x = x + embedded_positions
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
