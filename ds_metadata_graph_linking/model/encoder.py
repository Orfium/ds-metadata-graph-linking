import torch

from torch.nn import BatchNorm1d
from torch_geometric.nn import SAGEConv


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=(-1, -1),
                              out_channels=config.hidden_channels,
                              dropout=config.hidden_dropout_prob,
                              norm=BatchNorm1d(config.hidden_channels))
        self.conv2 = SAGEConv(in_channels=(-1, -1),
                              out_channels=config.out_channels,
                              dropout=config.hidden_dropout_prob,
                              norm=BatchNorm1d(config.out_channels))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
