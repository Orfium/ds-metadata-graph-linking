import click
import wandb
from torch_geometric.loader import LinkNeighborLoader

from ds_metadata_graph_linking.dataset.factory import create_dataloader, create_dataset, create_toy_dataset
from ds_metadata_graph_linking.trainer.config import load_config
from ds_metadata_graph_linking.trainer.trainer import Trainer
from ds_metadata_graph_linking.utils.edges import Edges
from ds_metadata_graph_linking.utils.train import set_seed

wandb.init(project="link_prediction", entity="stavros-giorgis", name="r2c_test")


@click.group()
def cli():
    pass


@cli.command(name='eval')
@click.option('--dataset_path', type=click.STRING, required=True)
@click.option('--checkpoints_path', type=click.STRING, required=True)
def val_entrypoint(dataset_path, checkpoints_path):
    import os
    import json
    import torch
    import numpy as np
    import pandas as pd

    from types import SimpleNamespace
    from ds_metadata_graph_linking.model.model import Model
    from ds_metadata_graph_linking.utils.infer import infer_predictions_from_logits

    metadata_file_path = os.path.join(checkpoints_path, 'model_metadata.json')
    with open(metadata_file_path, "rb") as metadata_file:
        metadata = json.load(metadata_file)
        config = SimpleNamespace(**metadata['train_config'])
        config.device = 'cpu'

    neg_edge_to_predict = ('composition', 'neg_embedded', 'recording')
    neg_embedded = pd.read_csv('data/R2C/negative_embedded_sample.csv')

    dataset = create_dataset(dataset_path, split='train').data
    edge_label_index = (Edges.edge_to_predict, dataset[Edges.edge_to_predict].edge_label_index)
    dataloader = create_dataloader(config=config,
                                   dataset=dataset,
                                   edge_label_index=edge_label_index,
                                   neg_sampling_ratio=config.neighbor_loader_neg_sampling_ratio)

    model_file = os.path.join(config.checkpoints_path, 'model.bin')
    model_state = torch.load(model_file, map_location='cpu')
    model = Model(config, dataset, dataloader)
    model.model.load_state_dict(model_state)

    edge_label_index = torch.from_numpy(neg_embedded.values).t().contiguous()

    logits = model(x_dict=dataset.x_dict,
                   edge_index_dict=dataset.edge_index_dict,
                   edge_label_index=edge_label_index).view(-1)

    logits = logits.detach().cpu()  # to stop tracking gradients
    predictions = infer_predictions_from_logits(logits).numpy()
    unique, counts = np.unique(predictions, return_counts=True)

    print(dict(zip(unique, counts)))


import torch
import numpy as np
import pandas as pd


@cli.command(name='train')
@click.option('--config', type=click.STRING, required=True)
@click.option('--dataset_path', type=click.STRING, required=True)
@click.option('--checkpoints_path', type=click.STRING, required=True)
def train_entrypoint(config, dataset_path, checkpoints_path):
    config = load_config(config, dataset_path, checkpoints_path)
    set_seed(config.seed)

    train_dataset = create_dataset(dataset_path, split='train').data
    train_edge_label_index = (Edges.edge_to_predict, train_dataset[Edges.edge_to_predict].edge_label_index)
    train_dataloader = create_dataloader(config=config,
                                         dataset=train_dataset,
                                         neg_sampling_ratio=0,  # no need to sample extra negative edges
                                         edge_label_index=train_edge_label_index,
                                         edge_label=train_dataset[Edges.edge_to_predict].edge_label)

    val_dataset = create_dataset(dataset_path, split='val').data
    val_edge_label_index = (Edges.edge_to_predict, val_dataset[Edges.edge_to_predict].edge_label_index)
    validation_dataloader = create_dataloader(config=config,
                                              dataset=val_dataset,
                                              neg_sampling_ratio=0,  # no need to sample extra negative edges
                                              edge_label_index=val_edge_label_index,
                                              edge_label=val_dataset[Edges.edge_to_predict].edge_label)

    trainer = Trainer(config, train_dataset, train_dataloader, validation_dataloader)
    trainer.train()


if __name__ == '__main__':
    cli()
