import click
import wandb

from ds_metadata_graph_linking.dataset.factory import create_dataloader, create_dataset, create_toy_dataset
from ds_metadata_graph_linking.trainer.config import load_config
from ds_metadata_graph_linking.trainer.trainer import Trainer
from ds_metadata_graph_linking.utils.edges import Edges
from ds_metadata_graph_linking.utils.train import set_seed

wandb.init(project="link_prediction", entity="stavros-giorgis", name="r2c_test")


@click.group()
def cli():
    pass


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
                                         edge_label_index=train_edge_label_index,
                                         neg_sampling_ratio=config.neighbor_loader_neg_sampling_ratio)

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
