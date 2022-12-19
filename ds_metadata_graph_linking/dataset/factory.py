import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

from torch_geometric.loader import LinkNeighborLoader

from ds_metadata_graph_linking.dataset.dataset import MetadataLinkingDataset
from ds_metadata_graph_linking.featurizers.fasttext import FastTextFeaturizer

from ds_metadata_graph_linking.utils.edges import Edges
from ds_metadata_graph_linking.utils.graph import sample_graph, generate_graph_statistics, \
    generate_indexes_for_relations, link_sample_graph


def create_toy_dataset():
    data = HeteroData()

    data['recording'].x = torch.rand((5, 128))
    data['composition'].x = torch.rand((3, 128))
    data['artist'].x = torch.rand((3, 128))
    data['client'].x = torch.rand((1, 128))

    data[('composition', 'embedded', 'recording')].edge_index = torch.tensor([[0, 2, 1], [1, 3, 4]])
    data[('composition', 'embedded', 'recording')].edge_label_index = torch.tensor([[0, 2, 1], [1, 3, 4]])
    data[('composition', 'embedded', 'recording')].edge_label = torch.tensor([1.0, 1.0, 1.0])

    data[('client', 'owns', 'composition')].edge_index = torch.tensor([[0, 0, 0], [0, 1, 2]])
    data[('artist', 'performed', 'recording')].edge_index = torch.tensor([[0, 0, 2, 1], [2, 1, 3, 0]])
    data[('artist', 'wrote', 'composition')].edge_index = torch.tensor([[1, 0], [1, 2]])

    transform = T.ToUndirected()
    data = transform(data)

    return data


def create_dataset(dataset_path, split):
    return MetadataLinkingDataset(dataset_path, split)


def create_dataloader(config, dataset, edge_label_index, neg_sampling_ratio=1, edge_label=None, shuffle=True):
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
    print('Loading raw data...')
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
    neg_embedded = pd.read_csv(osp.join(raw_data, 'neg_embedded.csv'))

    print(f'Sampling graph with base composition number {sample_size}')
    link_sample_graph(compositions=compositions, recordings=recordings, clients=clients, iswcs=iswcs, isrcs=isrcs,
                      embedded=embedded, neg_embedded=neg_embedded, performed=performed, wrote=wrote, has_isrc=has_isrc,
                      has_iswc=has_iswc, owns=owns, embedded_to_sample=sample_size, dataset_to_save=raw_graph_data)

    artists = pd.read_csv(osp.join(raw_graph_data, 'node-feat/artist_sample.csv'))
    recordings = pd.read_csv(osp.join(raw_graph_data, 'node-feat/recordings_sample.csv'))
    compositions = pd.read_csv(osp.join(raw_graph_data, 'node-feat/compositions_sample.csv'))
    clients = pd.read_csv(osp.join(raw_graph_data, 'node-feat/client_sample.csv'))
    iswcs = pd.read_csv(osp.join(raw_graph_data, 'node-feat/iswcs_sample.csv'))
    isrcs = pd.read_csv(osp.join(raw_graph_data, 'node-feat/isrcs_sample.csv'))

    embedded = pd.read_csv(osp.join(raw_graph_data, 'relations/embedded_sample.csv'))
    neg_embedded = pd.read_csv(osp.join(raw_graph_data, 'relations/neg_embedded_sample.csv'))
    has_isrc = pd.read_csv(osp.join(raw_graph_data, 'relations/has_isrc_sample.csv'))
    has_iswc = pd.read_csv(osp.join(raw_graph_data, 'relations/has_iswc_sample.csv'))
    owns = pd.read_csv(osp.join(raw_graph_data, 'relations/owns_sample.csv'))
    performed = pd.read_csv(osp.join(raw_graph_data, 'relations/performed_sample.csv'))
    wrote = pd.read_csv(osp.join(raw_graph_data, 'relations/wrote_sample.csv'))

    print(f'Generating indexes for relations')
    generate_indexes_for_relations(compositions=compositions, recordings=recordings, clients=clients, iswcs=iswcs,
                                   isrcs=isrcs, embedded=embedded, neg_embedded=neg_embedded,
                                   performed=performed, wrote=wrote, has_isrc=has_isrc,
                                   has_iswc=has_iswc, owns=owns, artists=artists, dataset_to_save=raw_graph_data)

    embedded = pd.read_csv(osp.join(raw_graph_data, 'relations/embedded_sample.csv'))
    neg_embedded = pd.read_csv(osp.join(raw_graph_data, 'relations/neg_embedded_sample.csv'))
    has_isrc = pd.read_csv(osp.join(raw_graph_data, 'relations/has_isrc_sample.csv'))
    has_iswc = pd.read_csv(osp.join(raw_graph_data, 'relations/has_iswc_sample.csv'))
    owns = pd.read_csv(osp.join(raw_graph_data, 'relations/owns_sample.csv'))
    performed = pd.read_csv(osp.join(raw_graph_data, 'relations/performed_sample.csv'))
    wrote = pd.read_csv(osp.join(raw_graph_data, 'relations/wrote_sample.csv'))

    print()
    print('Graph Statistics')
    generate_graph_statistics(compositions=compositions, recordings=recordings, clients=clients, iswcs=iswcs,
                              isrcs=isrcs, embedded=embedded, neg_embedded=neg_embedded,
                              performed=performed, wrote=wrote, has_isrc=has_isrc, has_iswc=has_iswc, owns=owns,
                              artists=artists)

    print('Generating Features for different nodes')
    featurizer = FastTextFeaturizer()
    featurizer.featurize(compositions=compositions, recordings=recordings,
                         clients=clients, artists=artists,
                         iswcs=iswcs, isrcs=isrcs, raw_graph_data=raw_graph_data)


def create_hetero_dataset_from_raw_graph_data(raw_graph_data, processed_data):
    data = HeteroData()

    print(f'Processing nodes...')
    process_nodes(data, raw_graph_data)

    print(f'Processing relations...')
    process_relations(data, raw_graph_data)

    transform = T.ToUndirected()
    data = transform(data)

    hetero_data_path = osp.join(processed_data, 'data.pt')
    torch.save(data, hetero_data_path)
    print(f'Hetero dataset was saved to {hetero_data_path}')


def negative_train_test_split(neg_edge_label_index, num_val, num_test):
    x_train, x_test = train_test_split(neg_edge_label_index.t(),
                                       test_size=num_val + num_test,
                                       random_state=42)
    x_test, x_val = train_test_split(x_test, test_size=num_test / (num_test + num_val))

    train_neg_edge_label_index = x_train.t()
    val_neg_edge_label_index = x_val.t()
    test_neg_edge_label_index = x_test.t()

    return train_neg_edge_label_index, val_neg_edge_label_index, test_neg_edge_label_index


def split_and_inject_negatives(raw_graph_data, train_data, num_val, val_data, num_test, test_data):
    neg_embedded = pd.read_csv(osp.join(raw_graph_data, 'relations', 'neg_embedded_sample.csv'))
    neg_edge_label_index = torch.from_numpy(neg_embedded[['compositions_index', 'recordings_index']].values).t()

    train_neg_edge_label_index, val_neg_edge_label_index, test_neg_edge_label_index = negative_train_test_split(
        neg_edge_label_index=neg_edge_label_index,
        num_val=num_val,
        num_test=num_test
    )

    train_neg_edge_label = torch.zeros_like(train_neg_edge_label_index[0])
    train_edge_label_index = torch.concat(dim=1, tensors=(train_data[Edges.edge_to_predict].edge_label_index,
                                                          train_neg_edge_label_index))
    train_edge_label = torch.concat((train_data[Edges.edge_to_predict].edge_label, train_neg_edge_label), dim=0)
    train_data[Edges.edge_to_predict].edge_label = train_edge_label
    train_data[Edges.edge_to_predict].edge_label_index = train_edge_label_index

    val_neg_edge_label = torch.zeros_like(val_neg_edge_label_index[0])
    val_edge_label_index = torch.concat(dim=1, tensors=(val_data[Edges.edge_to_predict].edge_label_index,
                                                        val_neg_edge_label_index))
    val_edge_label = torch.concat((val_data[Edges.edge_to_predict].edge_label, val_neg_edge_label), dim=0)
    val_data[Edges.edge_to_predict].edge_label = val_edge_label
    val_data[Edges.edge_to_predict].edge_label_index = val_edge_label_index

    test_neg_edge_label = torch.zeros_like(test_neg_edge_label_index[0])
    test_edge_label_index = torch.concat(dim=1, tensors=(test_data[Edges.edge_to_predict].edge_label_index,
                                                         test_neg_edge_label_index))
    test_edge_label = torch.concat((test_data[Edges.edge_to_predict].edge_label, test_neg_edge_label), dim=0)
    test_data[Edges.edge_to_predict].edge_label = test_edge_label
    test_data[Edges.edge_to_predict].edge_label_index = test_edge_label_index


def train_test_split_hetero_dataset(raw_graph_data, processed_data, num_val, num_test, neg_sampling_ratio,
                                    disjoint_train_ratio, add_negative_train_samples):
    print(f'Loading hetero dataset...')
    data = torch.load(osp.join(processed_data, 'data.pt'))

    train_data, val_data, test_data = T.RandomLinkSplit(num_val=num_val,
                                                        num_test=num_test,
                                                        is_undirected=True,
                                                        neg_sampling_ratio=neg_sampling_ratio,
                                                        disjoint_train_ratio=disjoint_train_ratio,
                                                        edge_types=Edges.edge_to_predict,
                                                        rev_edge_types=Edges.reverse_edge_to_predict,
                                                        add_negative_train_samples=add_negative_train_samples)(data)

    split_and_inject_negatives(raw_graph_data, train_data, num_val, val_data, num_test, test_data)

    torch.save(train_data, osp.join(processed_data, 'train_data.pt'))
    torch.save(val_data, osp.join(processed_data, 'val_data.pt'))
    torch.save(test_data, osp.join(processed_data, 'test_data.pt'))
    print(f'Hetero dataset train/val/test splits were saved to {processed_data}')


def process_nodes(data, raw_graph_data):
    print(f'Processing recording nodes...')
    recordings_path = osp.join(raw_graph_data, 'node-feat', 'recording_features.csv')
    recordings = pd.read_csv(recordings_path, header=None, dtype=np.float32)
    data['recording'].x = torch.from_numpy(recordings.values)

    print(f'Processing composition nodes...')
    compositions_path = osp.join(raw_graph_data, 'node-feat', 'composition_features.csv')
    compositions = pd.read_csv(compositions_path, header=None, dtype=np.float32)
    data['composition'].x = torch.from_numpy(compositions.values)

    print(f'Processing artist nodes...')
    artists_path = osp.join(raw_graph_data, 'node-feat', 'artist_features.csv')
    artists = pd.read_csv(artists_path, header=None, dtype=np.float32)
    data['artist'].x = torch.from_numpy(artists.values)

    print(f'Processing isrc nodes...')
    isrcs_path = osp.join(raw_graph_data, 'node-feat', 'isrcs_features.csv')
    isrcs = pd.read_csv(isrcs_path, header=None, dtype=np.float32)
    data['isrc'].x = torch.from_numpy(isrcs.values)

    print(f'Processing iswc nodes...')
    iswcs_path = osp.join(raw_graph_data, 'node-feat', 'iswcs_features.csv')
    iswcs = pd.read_csv(iswcs_path, header=None, dtype=np.float32)
    data['iswc'].x = torch.from_numpy(iswcs.values)

    print(f'Processing client nodes...')
    clients_path = osp.join(raw_graph_data, 'node-feat', 'clients_features.csv')
    clients = pd.read_csv(clients_path, header=None, dtype=np.float32)
    data['client'].x = torch.from_numpy(clients.values)


def process_relations(data, raw_graph_data, relations_dir='relations'):
    print(f'Processing composition_embedded_recording relation...')
    embedded = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'embedded_sample.csv'))
    data[('composition', 'embedded', 'recording')].edge_index = torch.from_numpy(embedded.values).t().contiguous()

    print(f'Processing recording_has_isrc_isrc relation...')
    has_isrc = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'has_isrc_sample.csv'))
    data[('recording', 'has_isrc', 'isrc')].edge_index = torch.from_numpy(has_isrc.values).t().contiguous()

    print(f'Processing composition_has_iswc_iswc relation...')
    has_iswc = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'has_iswc_sample.csv'))
    data[('composition', 'has_iswc', 'iswc')].edge_index = torch.from_numpy(has_iswc.values).t().contiguous()

    print(f'Processing client_owns_composition relation...')
    owns = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'owns_sample.csv'))
    data[('client', 'owns', 'composition')].edge_index = torch.from_numpy(owns.values).t().contiguous()

    print(f'Processing artist_performed_recording relation...')
    performed = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'performed_sample.csv'))
    data[('artist', 'performed', 'recording')].edge_index = torch.from_numpy(performed.values).t().contiguous()

    print(f'Processing artist_wrote_composition relation...')
    wrote = pd.read_csv(osp.join(raw_graph_data, relations_dir, 'wrote_sample.csv'))
    data[('artist', 'wrote', 'composition')].edge_index = torch.from_numpy(wrote.values).t().contiguous()
