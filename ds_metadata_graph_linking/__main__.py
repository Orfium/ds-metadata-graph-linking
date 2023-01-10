import json
import os
import os.path as osp

from types import SimpleNamespace

import click
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import f1_score, roc_curve, recall_score
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

from ds_metadata_graph_linking.dataset.factory import create_dataloader, create_dataset, create_toy_dataset
from ds_metadata_graph_linking.model.model import Model
from ds_metadata_graph_linking.trainer.config import load_config
from ds_metadata_graph_linking.trainer.trainer import Trainer
from ds_metadata_graph_linking.utils.edges import Edges
from ds_metadata_graph_linking.utils.infer import infer_predictions_from_logits
from ds_metadata_graph_linking.utils.train import set_seed, print_stats


@click.group()
def cli():
    pass


import fasttext as fastt

ft = fastt.load_model('cc.en.300.bin')

raw_data = 'data/base_dataset/test'
recordings = np.load(osp.join(raw_data, 'recordings.npz'), allow_pickle=True)['arr_0']
compositions = np.load(osp.join(raw_data, 'compositions.npz'), allow_pickle=True)['arr_0']
artists = np.load(osp.join(raw_data, 'artists.npz'), allow_pickle=True)['arr_0']
clients = np.load(osp.join(raw_data, 'clients.npz'), allow_pickle=True)['arr_0']
isrcs = np.load(osp.join(raw_data, 'isrcs.npz'), allow_pickle=True)['arr_0']
iswcs = np.load(osp.join(raw_data, 'iswcs.npz'), allow_pickle=True)['arr_0']


def generate_batch_embeddings(batch):
    recording_x = batch.x_dict['recording'].numpy()
    batch_recordings = recordings[recording_x]
    recording_title_features = []
    for batch_recording in batch_recordings:
        recording_title_features.append(ft.get_sentence_vector(str(batch_recording)))
    batch['recording'].x = torch.tensor(np.array(recording_title_features))

    composition_x = batch.x_dict['composition'].numpy()
    batch_compositions = compositions[composition_x]
    composition_title_features = []
    for batch_composition in batch_compositions:
        composition_title_features.append(ft.get_sentence_vector(str(batch_composition)))
    batch['composition'].x = torch.tensor(np.array(composition_title_features))

    artist_x = batch.x_dict['artist'].numpy()
    batch_artists = artists[artist_x]
    artist_title_features = []
    for batch_artist in batch_artists:
        artist_title_features.append(ft.get_sentence_vector(str(batch_artist)))
    batch['artist'].x = torch.tensor(np.array(artist_title_features))

    client_x = batch.x_dict['client'].numpy()
    batch_clients = clients[client_x]
    client_title_features = []
    for batch_client in batch_clients:
        client_title_features.append(ft.get_sentence_vector(str(batch_client)))
    batch['client'].x = torch.tensor(np.array(client_title_features))

    isrc_x = batch.x_dict['isrc'].numpy()
    batch_isrcs = isrcs[isrc_x]
    isrc_features = []
    for batch_isrc in batch_isrcs:
        isrc_features.append(ft.get_sentence_vector(str(batch_isrc)))
    batch['isrc'].x = torch.tensor(np.array(isrc_features))

    iswc_x = batch.x_dict['iswc'].numpy()
    batch_iswcs = iswcs[iswc_x]
    iswc_features = []
    for batch_iswc in batch_iswcs:
        iswc_features.append(ft.get_sentence_vector(str(batch_iswc)))
    batch['iswc'].x = torch.tensor(np.array(iswc_features))


@cli.command(name='eval')
@click.option('--dataset_path', type=click.STRING, required=True)
@click.option('--neg_embedded_path', type=click.STRING, required=True)
@click.option('--checkpoints_path', type=click.STRING, required=True)
def val_entrypoint(dataset_path, neg_embedded_path, checkpoints_path):
    metadata_file_path = os.path.join(checkpoints_path, 'model_metadata.json')
    with open(metadata_file_path, "rb") as metadata_file:
        metadata = json.load(metadata_file)
        config = SimpleNamespace(**metadata['train_config'])
        config.device = 'cpu'

    dataset = torch.load(dataset_path)
    print('Loaded dataset...')

    proposed_matches = pd.read_csv(os.path.join(raw_data, 'proposed_matches.csv'))
    print('Loaded proposed matches edges...')
    pairs = proposed_matches[['compositions_index', 'recordings_index']]
    labels = proposed_matches['STATUS'].tolist()

    dataset[Edges.edge_to_predict].edge_label_index = torch.from_numpy(pairs.values).t().contiguous()
    dataset[Edges.edge_to_predict].edge_label = torch.FloatTensor(labels)

    edge_label_index = (Edges.edge_to_predict, dataset[Edges.edge_to_predict].edge_label_index)
    dataloader = create_dataloader(config=config,
                                   dataset=dataset,
                                   edge_label_index=edge_label_index,
                                   neg_sampling_ratio=0)

    model_file = os.path.join(config.checkpoints_path, 'model.bin')
    model_state = torch.load(model_file, map_location='cpu')
    model = Model(config, dataset, dataloader)
    model.model.load_state_dict(model_state)

    fpr_scores = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, batch in progress_bar:
        generate_batch_embeddings(batch)

        edge_to_predict_storage = batch[Edges.edge_to_predict]
        edge_label = edge_to_predict_storage.edge_label
        edge_label_index = edge_to_predict_storage.edge_label_index

        logits = model(x_dict=batch.x_dict,
                       edge_index_dict=batch.edge_index_dict,
                       edge_label_index=edge_label_index).view(-1)

        logits = logits.detach().cpu()  # to stop tracking gradients
        predictions = infer_predictions_from_logits(logits).numpy()
        labels = edge_label.detach().cpu().numpy()
        print(predictions)
        print(labels)

        batch_fpr = 1 - recall_score(labels, predictions, pos_label=0)
        fpr_scores.append(batch_fpr)

        progress_bar.set_description(f'batch_fpr:{batch_fpr}')

    from statistics import mean
    print(f'avg fpr_score = {mean(fpr_scores)}')


@cli.command(name='train')
@click.option('--config', type=click.STRING, required=True)
@click.option('--dataset_path', type=click.STRING, required=True)
@click.option('--checkpoints_path', type=click.STRING, required=True)
def train_entrypoint(config, dataset_path, checkpoints_path):
    config = load_config(config, dataset_path, checkpoints_path)
    set_seed(config.seed)
    print(config)

    # wandb.init(project="link_prediction",
    #            entity="stavros-giorgis",
    #            config=config)

    train_dataset = create_dataset(dataset_path, split='train').data
    for index in range(train_dataset.x_dict['recording'].shape[0]):
        train_dataset.x_dict['recording'][index][1] = index

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

    print_stats(train_dataset, train_dataloader, val_dataset, validation_dataloader)

    trainer = Trainer(config, train_dataset, train_dataloader, validation_dataloader)
    trainer.train()


if __name__ == '__main__':
    cli()
