# models.py

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) for extracting features from graph-structured data.
    """
    def __init__(self, input_size, hidden_size, heads=8, dropout=0):
        super().__init__()
        self.conv1 = GATConv(input_size, hidden_size, heads=heads)
        self.conv2 = GATConv(hidden_size * heads, hidden_size, heads=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class MultiScaleCNN(nn.Module):
    """
    Multi-Scale Convolutional Neural Network (CNN) for sequential feature extraction.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.fc = nn.Linear(input_size * 32 * 3, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x1 = self.relu(self.conv1(x)).view(x.size(0), -1)
        x2 = self.relu(self.conv2(x)).view(x.size(0), -1)
        x3 = self.relu(self.conv3(x)).view(x.size(0), -1)
        x = torch.cat((x1, x2, x3), dim=1)  # Concatenate features
        x = self.relu(self.fc(x))
        return x


class TransformerEncoderModel(nn.Module):
    """
    Transformer Encoder for capturing global dependencies in sequential data.
    """
    def __init__(self, input_size, hidden_size, num_heads=2, num_layers=1, dropout=0.5):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.fc_input(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x).mean(dim=1)  # Aggregate across sequence
        x = self.layer_norm(x)
        x = self.relu(self.fc_output(x))
        x = self.dropout(x)
        return x


class FusionModel(nn.Module):
    """
    Fusion Model combining GAT, Multi-Scale CNN, and Transformer Encoder outputs.
    """
    def __init__(self, input_size, gat_hidden_size, cnn_hidden_size, transformer_hidden_size, output_size, dropout=0.7):
        super().__init__()
        self.gat_model = GATModel(input_size, gat_hidden_size)
        self.cnn_model = MultiScaleCNN(input_size, cnn_hidden_size)
        self.transformer_model = TransformerEncoderModel(input_size, transformer_hidden_size)
        self.fc = nn.Linear(gat_hidden_size + cnn_hidden_size + transformer_hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_data, X):
        gat_features = self.gat_model(graph_data)
        cnn_features = self.cnn_model(X)
        transformer_features = self.transformer_model(X)
        fused_features = torch.cat((gat_features, cnn_features, transformer_features), dim=1)
        fused_features = self.dropout(fused_features)
        out = self.fc(fused_features)
        return out
