"""
Fraud Detection Neural Network Models

This module contains various neural network architectures for fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class FraudDetectionNN(nn.Module):
    """
    Deep Neural Network for Fraud Detection

    A multi-layer feedforward neural network with dropout and batch normalization
    designed specifically for fraud detection tasks.
    """

    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64, 32],
                 output_size: int = 2,
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        """
        Initialize the fraud detection neural network

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes (2 for binary classification)
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(FraudDetectionNN, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build the network layers
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # Activation function
            layers.append(nn.ReLU())

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)

    def predict_proba(self, x):
        """Get prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs

    def predict(self, x):
        """Get class predictions"""
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
            return predictions


class FraudDetectionLSTM(nn.Module):
    """
    LSTM-based model for fraud detection with sequential features

    This model can be used when transaction data has temporal sequences.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 2,
                 dropout_rate: float = 0.3,
                 sequence_length: int = 10):
        """
        Initialize LSTM model for fraud detection

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            output_size: Number of output classes
            dropout_rate: Dropout probability
            sequence_length: Length of input sequences
        """
        super(FraudDetectionLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        """Forward pass through LSTM network"""
        # x shape: (batch_size, sequence_length, input_size)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last output from the sequence
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class FraudDetectionAutoencoder(nn.Module):
    """
    Autoencoder for anomaly-based fraud detection

    This model learns to reconstruct normal transactions and flags
    transactions with high reconstruction error as potential fraud.
    """

    def __init__(self,
                 input_size: int,
                 encoding_sizes: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2):
        """
        Initialize autoencoder for fraud detection

        Args:
            input_size: Number of input features
            encoding_sizes: List of layer sizes for encoder (decoder is mirrored)
            dropout_rate: Dropout probability
        """
        super(FraudDetectionAutoencoder, self).__init__()

        self.input_size = input_size
        self.encoding_sizes = encoding_sizes

        # Encoder
        encoder_layers = []
        prev_size = input_size

        for encoding_size in encoding_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, encoding_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = encoding_size

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        decoding_sizes = list(reversed(encoding_sizes[:-1])) + [input_size]

        for i, decoding_size in enumerate(decoding_sizes):
            decoder_layers.append(nn.Linear(prev_size, decoding_size))
            if i < len(decoding_sizes) - 1:  # No activation on final layer
                decoder_layers.extend([
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
            prev_size = decoding_size

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Get encoded representation"""
        return self.encoder(x)

    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
            return error


class FraudDetectionEnsemble(nn.Module):
    """
    Ensemble model combining multiple neural networks for fraud detection
    """

    def __init__(self,
                 input_size: int,
                 num_models: int = 3,
                 hidden_sizes: List[int] = [256, 128, 64],
                 output_size: int = 2,
                 dropout_rate: float = 0.3):
        """
        Initialize ensemble model

        Args:
            input_size: Number of input features
            num_models: Number of models in ensemble
            hidden_sizes: Hidden layer sizes for each model
            output_size: Number of output classes
            dropout_rate: Dropout probability
        """
        super(FraudDetectionEnsemble, self).__init__()

        self.num_models = num_models
        self.models = nn.ModuleList()

        # Create multiple models with slight variations
        for i in range(num_models):
            # Vary the architecture slightly for each model
            varied_hidden_sizes = [int(size * (0.8 + 0.4 * np.random.random()))
                                   for size in hidden_sizes]

            model = FraudDetectionNN(
                input_size=input_size,
                hidden_sizes=varied_hidden_sizes,
                output_size=output_size,
                dropout_rate=dropout_rate + 0.1 * np.random.random() - 0.05
            )
            self.models.append(model)

    def forward(self, x):
        """Forward pass through ensemble (average predictions)"""
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Average the outputs
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output

    def predict_proba(self, x):
        """Get ensemble prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs


def get_model(model_type: str, input_size: int, **kwargs):
    """
    Factory function to create different types of fraud detection models

    Args:
        model_type: Type of model ('nn', 'lstm', 'autoencoder', 'ensemble')
        input_size: Number of input features
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    model_type = model_type.lower()

    if model_type == 'nn' or model_type == 'neural_network':
        return FraudDetectionNN(input_size, **kwargs)
    elif model_type == 'lstm':
        return FraudDetectionLSTM(input_size, **kwargs)
    elif model_type == 'autoencoder':
        return FraudDetectionAutoencoder(input_size, **kwargs)
    elif model_type == 'ensemble':
        return FraudDetectionEnsemble(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    input_size = 14  # Based on our synthetic data

    print("Testing Fraud Detection Models:")
    print("=" * 50)

    # Test basic neural network
    print("1. Basic Neural Network:")
    model_nn = get_model('nn', input_size)
    print(f"   Model: {model_nn.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model_nn.parameters()):,}")

    # Test with dummy input
    dummy_input = torch.randn(32, input_size)
    output = model_nn(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print()

    # Test LSTM model
    print("2. LSTM Model:")
    model_lstm = get_model('lstm', input_size, sequence_length=5)
    print(f"   Model: {model_lstm.__class__.__name__}")
    print(
        f"   Parameters: {sum(p.numel() for p in model_lstm.parameters()):,}")

    # Test with sequence input
    dummy_sequence = torch.randn(32, 5, input_size)
    output_lstm = model_lstm(dummy_sequence)
    print(f"   Input shape: {dummy_sequence.shape}")
    print(f"   Output shape: {output_lstm.shape}")
    print()

    # Test Autoencoder
    print("3. Autoencoder Model:")
    model_ae = get_model('autoencoder', input_size)
    print(f"   Model: {model_ae.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model_ae.parameters()):,}")

    output_ae = model_ae(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output_ae.shape}")
    print()

    # Test Ensemble
    print("4. Ensemble Model:")
    model_ensemble = get_model('ensemble', input_size, num_models=3)
    print(f"   Model: {model_ensemble.__class__.__name__}")
    print(
        f"   Parameters: {sum(p.numel() for p in model_ensemble.parameters()):,}")

    output_ensemble = model_ensemble(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output_ensemble.shape}")
    print()

    print("✓ All models created successfully!")
