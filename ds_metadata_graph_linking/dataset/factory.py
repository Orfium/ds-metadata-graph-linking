import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from torch_geometric.loader import LinkNeighborLoader

from ds_metadata_graph_linking.dataset.dataset import MetadataLinkingDataset
from ds_metadata_graph_linking.featurizers.fasttext import FastTextFeaturizer

from ds_metadata_graph_linking.utils.edges import Edges
from ds_metadata_graph_linking.utils.graph import sample_graph, generate_graph_statistics, \
    generate_indexes_for_relations


def create_dataset(dataset_path, split):
    return MetadataLinkingDataset(dataset_path, split)


def create_dataloader(config, dataset, edge_label_index, neg_sampling_ratio, edge_label=None, shuffle=True):
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


def create_raw_graph_data_from_raw(sample_size: int, raw_data: str, raw_graph_data: str):
    print('Loadin raw data...')
    recordings = pd.read_csv(osp.join(raw_data, 'recording.csv'))
    recordings = recordings.dropna(subset=['recording_title'])
    compositions = pd.read_csv(osp.join(raw_data, 'composition.csv'))
    compositions = compositions.dropna(subset=['composition_title'])
    clients = pd.read_csv(osp.join(raw_data, 'client.csv'))
    iswcs = pd.read_csv(osp.join(raw_data, 'iswc.csv'))
    isrcs = pd.read_csv(osp.join(raw_data, 'isrc.csv'))

    embedded = pd.read_csv(osp.join(raw_data, 'embedded.csv'))
    has_isrc = pd.read_csv(osp.join(raw_data, 'has_isrc.csv'))
    has_iswc = pd.read_csv(osp.join(raw_data, 'has_iswc.csv'))
    owns = pd.read_csv(osp.join(raw_data, 'owns.csv'))
    performed = pd.read_csv(osp.join(raw_data, 'performed.csv'))
    wrote = pd.read_csv(osp.join(raw_data, 'wrote.csv'))

    print(f'Sampling graph with base composition number {sample_size}')
    sample_graph(compositions=compositions, recordings=recordings, clients=clients, iswcs=iswcs, isrcs=isrcs,
                 embedded=embedded, performed=performed, wrote=wrote, has_isrc=has_isrc, has_iswc=has_iswc, owns=owns,
                 compositions_to_sample=sample_size, dataset_to_save=raw_graph_data)

    artists = pd.read_csv(osp.join(raw_graph_data, 'nodes/artists_sample.csv'))
    recordings = pd.read_csv(osp.join(raw_graph_data, 'nodes/recordings_sample.csv'))
    compositions = pd.read_csv(osp.join(raw_graph_data, 'nodes/compositions_sample.csv'))
    clients = pd.read_csv(osp.join(raw_graph_data, 'nodes/client_sample.csv'))
    iswcs = pd.read_csv(osp.join(raw_graph_data, 'nodes/iswcs_sample.csv'))
    isrcs = pd.read_csv(osp.join(raw_graph_data, 'nodes/isrcs_sample.csv'))

    embedded = pd.read_csv(osp.join(raw_graph_data, 'relations/embedded_sample.csv'))
    has_isrc = pd.read_csv(osp.join(raw_graph_data, 'relations/has_isrc_sample.csv'))
    has_iswc = pd.read_csv(osp.join(raw_graph_data, 'relations/has_iswc_sample.csv'))
    owns = pd.read_csv(osp.join(raw_graph_data, 'relations/owns_sample.csv'))
    performed = pd.read_csv(osp.join(raw_graph_data, 'relations/performed_sample.csv'))
    wrote = pd.read_csv(osp.join(raw_graph_data, 'relations/wrote_sample.csv'))

    print(f'Generating indexes for relations')
    generate_indexes_for_relations(compositions=compositions, recordings=recordings, clients=clients, iswcs=iswcs,
                                   isrcs=isrcs, embedded=embedded, performed=performed, wrote=wrote, has_isrc=has_isrc,
                                   has_iswc=has_iswc, owns=owns, artists=artists, dataset_to_save=raw_graph_data)

    embedded = pd.read_csv(osp.join(raw_graph_data, 'relations/embedded_sample.csv'))
    has_isrc = pd.read_csv(osp.join(raw_graph_data, 'relations/has_isrc_sample.csv'))
    has_iswc = pd.read_csv(osp.join(raw_graph_data, 'relations/has_iswc_sample.csv'))
    owns = pd.read_csv(osp.join(raw_graph_data, 'relations/owns_sample.csv'))
    performed = pd.read_csv(osp.join(raw_graph_data, 'relations/performed_sample.csv'))
    wrote = pd.read_csv(osp.join(raw_graph_data, 'relations/wrote_sample.csv'))

    print()
    print('Graph Statistics')
    generate_graph_statistics(compositions=compositions, recordings=recordings, clients=clients, iswcs=iswcs,
                              isrcs=isrcs, embedded=embedded, performed=performed, wrote=wrote, has_isrc=has_isrc,
                              has_iswc=has_iswc, owns=owns, artists=artists)

    print('Generating Features for different nodes')
    featurizer = FastTextFeaturizer()
    featurizer.featurize(compositions=compositions, recordings=recordings,
                         clients=clients, artists=artists,
                         iswcs=iswcs, isrcs=isrcs, raw_graph_data=raw_graph_data)


def create_hetero_dataset_from_raw_graph_data(raw_graph_data, processed_data):
    data = HeteroData()

    process_nodes(data, raw_graph_data)
    process_relations(data, raw_graph_data)

    torch.save(data, osp.join(processed_data, 'data.pt'))


def train_test_split_hetero_dataset(processed_data, num_val, num_test, neg_sampling_ratio,
                                    disjoint_train_ratio, add_negative_train_samples):
    data = torch.load(osp.join(processed_data, 'data.pt'))

    train_data, val_data, test_data = T.RandomLinkSplit(num_val=num_val,
                                                        num_test=num_test,
                                                        is_undirected=True,
                                                        neg_sampling_ratio=neg_sampling_ratio,
                                                        disjoint_train_ratio=disjoint_train_ratio,
                                                        edge_types=Edges.edge_to_predict,
                                                        rev_edge_types=Edges.reverse_edge_to_predict,
                                                        add_negative_train_samples=add_negative_train_samples)(data)

    torch.save(train_data, osp.join(processed_data, 'train_data.pt'))
    torch.save(val_data, osp.join(processed_data, 'val_data.pt'))
    torch.save(test_data, osp.join(processed_data, 'test_data.pt'))


def process_nodes(data, raw_graph_data):
    recordings_path = osp.join(raw_graph_data, 'node-feat', 'recording_features.csv')
    recordings = pd.read_csv(recordings_path, header=None, dtype=np.float32)
    data['recording'].x = torch.from_numpy(recordings.values)

    compositions_path = osp.join(raw_graph_data, 'node-feat', 'composition_features.csv')
    compositions = pd.read_csv(compositions_path, header=None, dtype=np.float32)
    data['composition'].x = torch.from_numpy(compositions.values)

    artists_path = osp.join(raw_graph_data, 'node-feat', 'artist_features.csv')
    artists = pd.read_csv(artists_path, header=None, dtype=np.float32)
    data['artist'].x = torch.from_numpy(artists.values)

    isrcs_path = osp.join(raw_graph_data, 'node-feat', 'isrcs_features.csv')
    isrcs = pd.read_csv(isrcs_path, header=None, dtype=np.float32)
    data['isrc'].x = torch.from_numpy(isrcs.values)

    iswcs_path = osp.join(raw_graph_data, 'node-feat', 'iswcs_features.csv')
    iswcs = pd.read_csv(iswcs_path, header=None, dtype=np.float32)
    data['iswc'].x = torch.from_numpy(iswcs.values)

    clients_path = osp.join(raw_graph_data, 'node-feat', 'clients_features.csv')
    clients = pd.read_csv(clients_path, header=None, dtype=np.float32)
    data['client'].x = torch.from_numpy(clients.values)


def process_relations(data, raw_graph_data, relations_dir='relations'):
    embedded = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'embedded_sample.csv'))
    data[('composition', 'embedded', 'recording')].edge_index = torch.from_numpy(embedded.values).t().contiguous()

    has_isrc = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'has_isrc_sample.csv'))
    data[('recording', 'has_isrc', 'isrc')].edge_index = torch.from_numpy(has_isrc.values).t().contiguous()

    has_iswc = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'has_iswc_sample.csv'))
    data[('composition', 'has_iswc', 'iswc')].edge_index = torch.from_numpy(has_iswc.values).t().contiguous()

    owns = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'owns_sample.csv'))
    data[('client', 'owns', 'composition')].edge_index = torch.from_numpy(owns.values).t().contiguous()

    performed = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'performed_sample.csv'))
    data[('artist', 'performed', 'recording')].edge_index = torch.from_numpy(performed.values).t().contiguous()

    wrote = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'wrote_sample.csv'))
    data[('artist', 'wrote', 'composition')].edge_index = torch.from_numpy(wrote.values).t().contiguous()
