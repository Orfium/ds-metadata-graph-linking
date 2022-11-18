import yaml
import pprint

from ds_metadata_graph_linking.utils.device import decide_device, Devices


class TrainConfig:
    def __init__(self, **kwargs):
        self.seed = kwargs['train'].pop('seed', 666)
        self.resume = kwargs['train'].pop('resume', False)
        self.patience = kwargs['train'].pop('patience', 10)
        self.epochs = kwargs['train'].pop('num_epochs', 100)
        self.batch_size = kwargs['train'].pop('batch_size', 1)
        self.model_log_freq = kwargs['train'].pop('model_log_freq', 100)
        self.log_every_n_steps = kwargs['train'].pop('log_every_n_steps', 10)

        self.num_workers = kwargs['train'].pop('num_workers', 1)
        self.num_neighbors = kwargs['train'].pop('num_neighbors', [-1, -1])
        self.neighbor_loader_neg_sampling_ratio = kwargs['train'].pop('neighbor_loader_neg_sampling_ratio', 0)

        self.architecture = kwargs['model'].pop('architecture', 'gnn')
        self.num_labels = kwargs['model'].pop('num_labels', 1)
        self.aggr = kwargs['model'].pop('aggr', 'sum')
        self.out_channels = kwargs['model'].pop('out_channels', 64)
        self.hidden_channels = kwargs['model'].pop('hidden_channels', 64)
        self.hidden_dropout_prob = kwargs['model'].pop('hidden_dropout_prob', 0.2)

        self.optim = kwargs['optimizer'].pop('optim', 'adam')
        self.beta1 = kwargs['optimizer'].pop('beta1', 0.9)
        self.beta2 = kwargs['optimizer'].pop('beta2', 0.999)
        self.epsilon = float(kwargs['optimizer'].pop('epsilon', 1e-8))
        self.learning_rate = float(kwargs['optimizer'].pop('lr', 0.001))

        self.device = decide_device(kwargs['train'].pop('device', Devices.GPU))

        self.dataset_path = kwargs['train']['dataset_path']
        self.checkpoints_path = kwargs['train']['checkpoints_path']

    def __repr__(self):
        return pprint.pformat(self.__dict__)


def load_config(path, dataset_path, checkpoints_path):
    with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    config['train']['dataset_path'] = dataset_path
    config['train']['checkpoints_path'] = checkpoints_path

    config = TrainConfig(**config)

    return config
