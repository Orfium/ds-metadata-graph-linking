import torch
import random

import numpy as np

from ds_metadata_graph_linking.utils.edges import Edges


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def print_stats(train_dataset, train_dataloader, val_dataset, val_dataloader):
    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))

    print('train_dataset', train_dataset)
    print('val_dataset', val_dataset)

    print('train_batch', train_batch)
    print('val_batch', val_batch)

    print('train_batch_stats', np.unique(train_batch[Edges.edge_to_predict].edge_label.numpy(), return_counts=True))
    print('val_batch_stats', np.unique(val_batch[Edges.edge_to_predict].numpy(), return_counts=True))
