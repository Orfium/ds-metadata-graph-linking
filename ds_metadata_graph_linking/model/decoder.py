import torch

from torch_geometric.nn import Linear


class Decoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = Linear(2 * config.hidden_channels, config.out_channels)
        self.lin2 = Linear(config.out_channels, config.num_labels)

    def forward(self, z_dict, edge_label_index):
        z = torch.cat([z_dict['composition'][edge_label_index[0]],
                       z_dict['recording'][edge_label_index[1]]], dim=-1)
        z = self.lin1(z).relu()
        return self.lin2(z)
