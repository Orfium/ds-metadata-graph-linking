from ds_metadata_graph_linking.model.gnn import GNN
from ds_metadata_graph_linking.utils.edges import Edges


class Model:
    def __init__(self, config, dataset, dataloader):
        self.config = config
        self.model = GNN(config, dataset)
        # self.lazy_initialization(dataloader)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.model = self.model.to(device)

    def train(self):
        self.model.to(self.config.device)
        self.model.train()

    def eval(self):
        self.model.eval()

    def state(self):
        return self.model.state_dict()

    def parameters(self):
        return [param for _, param in self.model.named_parameters() if param.requires_grad]

    def lazy_initialization(self, dataloader):
        self.to(self.config.device)

        # lazy initialization of model parameters
        batch = next(iter(dataloader))
        batch = batch.to(self.config.device)
        self.model(batch.x_dict, batch.edge_index_dict, batch[Edges.edge_to_predict].edge_label_index)
