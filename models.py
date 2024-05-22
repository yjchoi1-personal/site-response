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
    """
    A 1D convolutional neural network module for sequence processing.

    Attributes:
        sequence_length (int): Length of the input sequences.
        n_features (int): Number of features per time step in the input.
        out_channels (list of int): List of output channels for each convolutional layer.
        kernel_sizes (list of int): List of kernel sizes for each convolutional layer.
        pool_sizes (list of int): List of pool sizes for each pooling layer after convolution.

    """

    def __init__(
            self,
            sequence_length,
            n_features,
            out_channels=[16, 32, 16],
            kernel_sizes=[24, 12, 6],
            pool_sizes=[2, 2, 2]
    ):
        super(Conv1D, self).__init__()

        if not (len(out_channels) == len(kernel_sizes) == len(pool_sizes)):
            raise ValueError("The length of out_channels, kernel_sizes, and pool_sizes must be the same.")

        n_conv_layers = len(out_channels)
        conv_layers = []
        current_channels = n_features

        for i in range(n_conv_layers):
            conv = nn.Conv1d(in_channels=current_channels, out_channels=out_channels[i],
                             kernel_size=kernel_sizes[i])
            relu = nn.ReLU()
            pool = nn.MaxPool1d(kernel_size=pool_sizes[i])
            conv_layers += [conv, relu, pool]
            current_channels = out_channels[i]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        # Calculating the output dimensions after convolutional and pooling layers
        dummy_input = torch.randn(1, n_features, sequence_length)
        dummy_output = self.conv_layers(dummy_input)
        self.flat_features = dummy_output.numel()

        # Dense layer to output the original sequence length
        self.dense = nn.Linear(self.flat_features, sequence_length)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data tensor (batch_size, n_features, sequence_length).

        Returns:
            torch.Tensor: The output data tensor after passing through the convolutional layers,
                          being flattened and processed by a dense and activation layer.
        """
        # Ensure input tensor is in the correct shape: (batch_size, n_features, sequence_length)
        x = x.permute(0, 2, 1)

        x = self.conv_layers(x)  # Pass through convolutional layers
        x = self.flatten(x)      # Flatten the output for the dense layer
        x = self.dense(x)        # Dense layer processing
        x = self.softplus(x)     # Activation function

        return x


# class EncodeDecodeCNN(nn.Module):
#     def __init__(
#             self,
#     ):


class SequenceLSTM(nn.Module):
    def __init__(
            self,
            sequence_length,
            n_features,
            hidden_dim=32,
            n_lstm_layers=3):
        super(SequenceLSTM, self).__init__()
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

        # # MLP
        # self.mlp = build_mlp(
        #     self.flat_features, [mlp_hidden_dim for _ in range(nmlp_layers)], output_size=sequence_length)
        # self.layer_norm = nn.LayerNorm(sequence_length)

    def forward(self, x):

        # Process through LSTM layers
        x, _ = self.lstm(x)

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
            dropout=0,
            positional_encoding=False
    ):

        super(TimeSeriesTransformer2, self).__init__()
        self.sequence_length = sequence_length
        self.n_input_features = n_input_features
        self.embedding_size = embedding_size
        self.positional_encoding = positional_encoding

        # Embeddings for features and periods
        self.feature_embedding = nn.Linear(n_input_features, embedding_size)

        # Positional encoding for adding notion of time step
        if self.positional_encoding:
            self.positional_encoder = PositionalEncoding(
                sequence_length, embedding_size)

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

        # Add positional encoding
        if self.positional_encoding:
            x = self.positional_encoder(x)

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
            embedding_size=16,  # 16
            nhead=4,  # 2
            num_encoder_layers=3,  # 3
            num_decoder_layers=6,  # 6
            dim_feedforward=2048,  # 2048
            dropout=0.1,  # 0.1
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
            num_decoder_layers=num_decoder_layers,
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
