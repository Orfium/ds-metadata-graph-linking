class Model:
    def __init__(self, config, dataset, dataloader):
        self.config = config
        self.model = ModelRegistry.get_model('gnn', dataset, dataloader)

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
