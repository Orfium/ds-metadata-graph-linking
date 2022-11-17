import click
import wandb

from ds_metadata_graph_linking.dataset.factory import create_dataloader, create_dataset, split_data
from ds_metadata_graph_linking.model.manager import ModelManager
from ds_metadata_graph_linking.trainer.config import load_config
from ds_metadata_graph_linking.trainer.criterion import CriterionManager
from ds_metadata_graph_linking.trainer.optimizer import OptimizerManager
from ds_metadata_graph_linking.trainer.trainer import train
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
@click.option('--debug/--no-debug', type=click.BOOL, default=False)
def train_entrypoint(config, dataset_path, checkpoints_path, debug):
    config = load_config(config, dataset_path, checkpoints_path)
    set_seed(config.seed)

    dataset = create_dataset(dataset_path, debug=debug)

    train_data, val_data, test_data = split_data(config, dataset)

    train_edge_label_index = (Edges.edge_to_predict, train_data[Edges.edge_to_predict].edge_label_index)
    train_dataloader = create_dataloader(config=config,
                                         dataset=train_data,
                                         edge_label_index=train_edge_label_index,
                                         neg_sampling_ratio=config.neighbor_loader_neg_sampling_ratio)

    val_edge_label_index = (Edges.edge_to_predict, val_data[Edges.edge_to_predict].edge_label_index)
    val_dataloader = create_dataloader(config=config,
                                       dataset=val_data,
                                       neg_sampling_ratio=0,  # no need to sample extra negative edges
                                       edge_label_index=val_edge_label_index,
                                       edge_label=val_data[Edges.edge_to_predict].edge_label)

    model = ModelManager(config, train_data, train_dataloader)
    optimizer = OptimizerManager(config, train_dataloader, model_manager=model)
    criterion = CriterionManager(config)

    train(config, train_dataloader, val_dataloader, val_data, test_data,
          model, optimizer, criterion, checkpoints_path)


if __name__ == '__main__':
    cli()
