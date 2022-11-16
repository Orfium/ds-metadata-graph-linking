import yaml
import pprint

from ds_metadata_graph_linking.utils.device import decide_device, Devices


class TrainConfig:
    def __init__(self, **kwargs):
        self.seed = kwargs['train'].pop('seed', 666)
        self.epochs = kwargs['train'].pop('num_epochs', 100)
        self.batch_size = kwargs['train'].pop('batch_size', 1)
        self.log_every = kwargs['train'].pop('log_every', 10)
        self.model_log_freq = kwargs['train'].pop('model_log_freq', 100)

        self.num_val = kwargs['data'].pop('num_val', 0)
        self.num_test = kwargs['data'].pop('num_test', 0)
        self.num_workers = kwargs['data'].pop('num_workers', 1)
        self.num_neighbors = kwargs['data'].pop('num_neighbors', [-1, -1])
        self.neg_sampling_ratio = kwargs['data'].pop('neg_sampling_ratio', 0)
        self.disjoint_train_ratio = kwargs['data'].pop('disjoint_train_ratio', 0)
        self.neighbor_loader_neg_sampling_ratio = kwargs['data'].pop('neighbor_loader_neg_sampling_ratio', 0)

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

    def __repr__(self):
        return pprint.pformat(self.__dict__)


def load_config(path, dataset_path, checkpoints_path):
    with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    config['train']['dataset_path'] = dataset_path
    config['train']['checkpoints_path'] = checkpoints_path

    config = TrainConfig(**config)

    return config
