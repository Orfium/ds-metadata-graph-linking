import torch

from torch_geometric.nn import to_hetero

from ds_metadata_graph_linking.model.decoder import Decoder
from ds_metadata_graph_linking.model.encoder import Encoder


class Model(torch.nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.encoder = to_hetero(Encoder(config), data.metadata(), aggr=config.aggr)
        self.decoder = Decoder(config)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
