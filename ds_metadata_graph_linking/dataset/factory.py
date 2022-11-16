import os
import torch
import numpy as np
import pandas as pd
import torch_geometric.transforms as T

from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from ds_metadata_graph_linking.utils.graph_utils import sample_graph, generate_graph_statistics, \
    generate_indexes_for_relations
from ds_metadata_graph_linking.utils.generate_features import generate_features_recordings, generate_features_artists, \
    generate_features_clients, generate_features_compositions, generate_features_isrcs, generate_features_iswcs


def create_toy_r2c_dataset():
    data = HeteroData()

    data['recording'].x = torch.rand((5, 128))
    data['composition'].x = torch.rand((3, 128))
    data['artist'].x = torch.rand((3, 128))
    data['client'].x = torch.rand((1, 128))

    data[('composition', 'embedded', 'recording')].edge_index = torch.tensor([[0, 2, 1], [1, 3, 4]])
    data[('client', 'owns', 'composition')].edge_index = torch.tensor([[0, 0, 0], [0, 1, 2]])
    data[('artist', 'performed', 'recording')].edge_index = torch.tensor([[0, 0, 2, 1], [2, 1, 3, 0]])
    data[('artist', 'wrote', 'composition')].edge_index = torch.tensor([[1, 0], [1, 2]])

    transform = T.ToUndirected()
    data = transform(data)

    return data


def create_dataset(dataset_path, debug):
    if debug:
        return create_toy_r2c_dataset()

    data = HeteroData()

    # read node types
    recordings = pd.read_csv(os.path.join(dataset_path, 'recording_features.csv'), header=None, dtype=np.float32)
    compositions = pd.read_csv(os.path.join(dataset_path, 'composition_features.csv'), header=None, dtype=np.float32)
    artists = pd.read_csv(os.path.join(dataset_path, 'artist_features.csv'), header=None, dtype=np.float32)
    isrcs = pd.read_csv(os.path.join(dataset_path, 'isrcs_features.csv'), header=None, dtype=np.float32)
    iswcs = pd.read_csv(os.path.join(dataset_path, 'iswcs_features.csv'), header=None, dtype=np.float32)
    clients = pd.read_csv(os.path.join(dataset_path, 'clients_features.csv'), header=None, dtype=np.float32)

    # read relations and edges
    embedded = pd.read_csv(os.path.join(dataset_path, 'embedded_sample.csv'))
    has_isrc = pd.read_csv(os.path.join(dataset_path, 'has_isrc_sample.csv'))
    has_iswc = pd.read_csv(os.path.join(dataset_path, 'has_iswc_sample.csv'))
    owns = pd.read_csv(os.path.join(dataset_path, 'owns_sample.csv'))
    performed = pd.read_csv(os.path.join(dataset_path, 'performed_sample.csv'))
    wrote = pd.read_csv(os.path.join(dataset_path, 'wrote_sample.csv'))

    # load to HeteroData object
    data['recording'].x = torch.from_numpy(recordings.values)
    data['composition'].x = torch.from_numpy(compositions.values)
    data['artist'].x = torch.from_numpy(artists.values)
    data['isrc'].x = torch.from_numpy(isrcs.values)
    data['iswc'].x = torch.from_numpy(iswcs.values)
    data['client'].x = torch.from_numpy(clients.values)

    # load to HeteroData object
    data[('composition', 'embedded', 'recording')].edge_index = torch.from_numpy(embedded.values).t().contiguous()
    data[('recording', 'has_isrc', 'isrc')].edge_index = torch.from_numpy(has_isrc.values).t().contiguous()
    data[('composition', 'has_iswc', 'iswc')].edge_index = torch.from_numpy(has_iswc.values).t().contiguous()
    data[('client', 'owns', 'composition')].edge_index = torch.from_numpy(owns.values).t().contiguous()
    data[('artist', 'performed', 'recording')].edge_index = torch.from_numpy(performed.values).t().contiguous()
    data[('artist', 'wrote', 'composition')].edge_index = torch.from_numpy(wrote.values).t().contiguous()

    transform = T.ToUndirected()
    data = transform(data)

    return data


def print_dataset_info(dataset, train_data, val_data, test_data, edge_to_predict):
    print('========================================== Dataset Stats ==========================================')
    print('********************************************* Dataset *********************************************')
    print('Dataset metadata: ', dataset.metadata)
    print(f'Nodes: {dataset.num_nodes:_}')
    print(f'Edges: {dataset.num_edges:_}')

    print('********************************************* Train Data *********************************************')
    print('Train metadata: ', train_data.metadata)
    print(f'Nodes: {train_data.num_nodes:_}')
    print(f'Edges: {train_data.num_edges:_}')
    print('train_data_edge_index: ', train_data[edge_to_predict].edge_index.shape)
    print('train_data_edge_label_index: ', train_data[edge_to_predict].edge_label_index.shape)

    print('********************************************* Val Data *********************************************')
    print('Val metadata: ', val_data.metadata)
    print(f'Nodes: {val_data.num_nodes:_}')
    print(f'Edges: {val_data.num_edges:_}')
    print('val_data_edge_index: ', val_data[edge_to_predict].edge_index.shape)
    print('val_data_edge_label_index: ', val_data[edge_to_predict].edge_label_index.shape)
    print('val_data_edge_label_value_counts: ', torch.bincount(val_data[edge_to_predict].edge_label.to(torch.long)))

    print('********************************************* Test Data *********************************************')
    print('Test metadata: ', test_data.metadata)
    print(f'Nodes: {test_data.num_nodes:_}')
    print(f'Edges: {test_data.num_edges:_}')
    print('test_data_edge_index: ', test_data[edge_to_predict].edge_index.shape)
    print('test_data_edge_label_index: ', test_data[edge_to_predict].edge_label_index.shape)
    print('test_data_edge_label_value_counts: ', torch.bincount(test_data[edge_to_predict].edge_label.to(torch.long)))

    print('========================================== Dataset Stats End ==========================================')


def print_batch_info(train_dataloader, val_dataloader, edge_to_predict):
    batch = next(iter(train_dataloader))
    print('========================================== Train Batch Stats ==========================================')

    print('Batch metadata: ', batch.metadata)
    print(f'Nodes: {batch.num_nodes:_}')
    print(f'Edges: {batch.num_edges:_}')
    print('batch_edge_index: ', batch[edge_to_predict].edge_index.shape)
    print('batch_edge_label_index: ', batch[edge_to_predict].edge_label_index.shape)
    print('batch_edge_label_value_counts: ', torch.bincount(batch[edge_to_predict].edge_label.to(torch.long)))

    print('========================================== Train Batch Stats End ==========================================')

    batch = next(iter(val_dataloader))
    print('========================================== Val Batch Stats ==========================================')

    print('Batch metadata: ', batch.metadata)
    print(f'Nodes: {batch.num_nodes:_}')
    print(f'Edges: {batch.num_edges:_}')
    print('batch_edge_index: ', batch[edge_to_predict].edge_index.shape)
    print('batch_edge_label_index: ', batch[edge_to_predict].edge_label_index.shape)
    print('batch_edge_label_value_counts: ', torch.bincount(batch[edge_to_predict].edge_label.to(torch.long)))

    print('========================================== Val Batch Stats End ==========================================')


def check_data(dataset, train_data, val_data, test_data, edge_to_predict):
    train_data_edge_label_index = set((tuple(i) for i in train_data[edge_to_predict].edge_label_index.t().numpy()))
    train_data_edge_index = set((tuple(i) for i in train_data[edge_to_predict].edge_index.t().numpy()))
    train_data_edge_index_intersection = len(train_data_edge_label_index.intersection(train_data_edge_index))
    print('train edge indexes intersection: ', train_data_edge_index_intersection / len(train_data_edge_label_index))

    train_data_e_set = set((tuple(i) for i in train_data[edge_to_predict].edge_label_index.t().numpy()))
    val_data_e_set = set((tuple(i) for i in val_data[edge_to_predict].edge_label_index.t().numpy()))
    test_data_e_set = set((tuple(i) for i in test_data[edge_to_predict].edge_label_index.t().numpy()))
    train_val_data_e_set_intersection = len(train_data_e_set.intersection(val_data_e_set))
    train_test_data_e_set_intersection = len(train_data_e_set.intersection(test_data_e_set))
    val_test_data_e_set_intersection = len(val_data_e_set.intersection(test_data_e_set))

    print('train-val edge_label_index intersection: ', train_val_data_e_set_intersection / len(val_data_e_set))
    print('train-test edge_label_index intersection: ', train_test_data_e_set_intersection / len(test_data_e_set))
    print('val-test edge_label_index intersection: ', val_test_data_e_set_intersection / len(test_data_e_set))

    train_negative_edges_indices = torch.where(train_data[edge_to_predict].edge_label == 0)[0]
    val_negative_edges_indices = torch.where(val_data[edge_to_predict].edge_label == 0)[0]
    test_negative_edges_indices = torch.where(test_data[edge_to_predict].edge_label == 0)[0]

    val_negative_edges = val_data[edge_to_predict].edge_label_index[:, val_negative_edges_indices]
    test_negative_edges = test_data[edge_to_predict].edge_label_index[:, test_negative_edges_indices]

    val_negative_edges = set((tuple(i) for i in val_negative_edges.t().numpy()))
    test_negative_edges = set((tuple(i) for i in test_negative_edges.t().numpy()))
    hetero_data_positive_edges = set((tuple(i) for i in dataset[edge_to_predict].edge_index.t().numpy()))

    val_data_e_set_intersection = len(val_negative_edges.intersection(hetero_data_positive_edges))
    test_data_e_set_intersection = len(test_negative_edges.intersection(hetero_data_positive_edges))

    print('val false negative edge_label_index: ', val_data_e_set_intersection / len(val_negative_edges))
    print('test false negative edge_label_index: ', test_data_e_set_intersection / len(test_negative_edges))


def split_data(config, dataset, edge_to_predict, reverse_edge_to_predict):
    train_data, val_data, test_data = T.RandomLinkSplit(num_val=config.num_val,
                                                        num_test=config.num_test,
                                                        is_undirected=True,
                                                        add_negative_train_samples=False,
                                                        neg_sampling_ratio=config.neg_sampling_ratio,
                                                        disjoint_train_ratio=config.disjoint_train_ratio,
                                                        edge_types=[edge_to_predict],
                                                        rev_edge_types=[reverse_edge_to_predict])(dataset)
    return train_data, val_data, test_data


def create_dataloader(config, dataset, edge_label_index, edge_label, neg_sampling_ratio, shuffle=True):
    return LinkNeighborLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.batch_size,
        neg_sampling_ratio=neg_sampling_ratio,
        num_neighbors=config.num_neighbors,
        edge_label=edge_label,
        edge_label_index=edge_label_index,
        num_workers=config.num_workers, persistent_workers=True
    )


def create_graph_dataset_from_raw(sample_size: int):

    print('Loading raw data')
    # entities

    artists = pd.read_csv('data/base_dataset/artist.csv')
    artists = artists.dropna(subset=['name'])
    recordings = pd.read_csv('data/base_dataset/recording.csv')
    recordings = recordings.dropna(subset=['recording_title'])
    compositions = pd.read_csv('data/base_dataset/composition.csv')
    compositions = compositions.dropna(subset=['composition_title'])
    clients = pd.read_csv('data/base_dataset/client.csv')
    iswcs = pd.read_csv('data/base_dataset/iswc.csv')
    isrcs = pd.read_csv('data/base_dataset/isrc.csv')

    # relationships

    embedded = pd.read_csv('data/base_dataset/embedded.csv')
    has_isrc = pd.read_csv('data/base_dataset/has_isrc.csv')
    has_iswc = pd.read_csv('data/base_dataset/has_iswc.csv')
    owns = pd.read_csv('data/base_dataset/owns.csv')
    performed = pd.read_csv('data/base_dataset/performed.csv')
    wrote = pd.read_csv('data/base_dataset/wrote.csv')

    print(f'Sampling graph with base composition number {sample_size}')
    sample_graph(compositions=compositions, recordings=recordings, clients=clients, iswcs=iswcs, isrcs=isrcs,
                 embedded=embedded, performed=performed, wrote=wrote, has_isrc=has_isrc, has_iswc=has_iswc, owns=owns,
                 compositions_to_sample=sample_size)

    # entities

    artists = pd.read_csv('data/base_dataset/processed_dataset/artists_sample.csv')
    recordings = pd.read_csv('data/base_dataset/processed_dataset/recordings_sample.csv')
    compositions = pd.read_csv('data/base_dataset/processed_dataset/compositions_sample.csv')
    clients = pd.read_csv('data/base_dataset/processed_dataset/client_sample.csv')
    iswcs = pd.read_csv('data/base_dataset/processed_dataset/iswcs_sample.csv')
    isrcs = pd.read_csv('data/base_dataset/processed_dataset/isrcs_sample.csv')

    # relationships

    embedded = pd.read_csv('data/base_dataset/processed_dataset/embedded_sample.csv')
    has_isrc = pd.read_csv('data/base_dataset/processed_dataset/has_isrc_sample.csv')
    has_iswc = pd.read_csv('data/base_dataset/processed_dataset/has_iswc_sample.csv')
    owns = pd.read_csv('data/base_dataset/processed_dataset/owns_sample.csv')
    performed = pd.read_csv('data/base_dataset/processed_dataset/performed_sample.csv')
    wrote = pd.read_csv('data/base_dataset/processed_dataset/wrote_sample.csv')

    print(f'Generating indexes for relations')
    generate_indexes_for_relations(compositions=compositions, recordings=recordings, clients=clients, iswcs=iswcs,
                                   isrcs=isrcs, embedded=embedded, performed=performed, wrote=wrote, has_isrc=has_isrc,
                                   has_iswc=has_iswc, owns=owns, artists=artists)

    # relationships

    embedded = pd.read_csv('data/base_dataset/processed_dataset/embedded_sample.csv')
    has_isrc = pd.read_csv('data/base_dataset/processed_dataset/has_isrc_sample.csv')
    has_iswc = pd.read_csv('data/base_dataset/processed_dataset/has_iswc_sample.csv')
    owns = pd.read_csv('data/base_dataset/processed_dataset/owns_sample.csv')
    performed = pd.read_csv('data/base_dataset/processed_dataset/performed_sample.csv')
    wrote = pd.read_csv('data/base_dataset/processed_dataset/wrote_sample.csv')

    print()
    print('Graph Statistics')
    generate_graph_statistics(compositions=compositions, recordings=recordings, clients=clients, iswcs=iswcs,
                              isrcs=isrcs, embedded=embedded, performed=performed, wrote=wrote, has_isrc=has_isrc,
                              has_iswc=has_iswc, owns=owns, artists=artists)

    print()
    print('Generating Features for different nodes')
    print('Recordings')
    generate_features_recordings(recordings)
    print('Compositions')
    generate_features_compositions(compositions)
    print('Iswcs')
    generate_features_iswcs(iswcs)
    print('Isrcs')
    generate_features_isrcs(isrcs)
    print('Artists')
    generate_features_artists(artists)
    print('Clients')
    generate_features_clients(clients)
