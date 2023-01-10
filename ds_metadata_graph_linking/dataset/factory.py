import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch_geometric.transforms as T
from numpy import savez_compressed
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


def test(raw_data, recordings, compositions, artists, clients, iswcs, isrcs,
         embedded, has_isrc, has_iswc, owns, performed, wrote, proposed_matches):
    data = HeteroData()

    from tqdm import tqdm

    for index, row in tqdm(proposed_matches.iterrows()):
        # check recording
        recording_id = row['RECORDING_ID']
        recording_isrc = row['ASSET_ISRC']
        recording_artists = row['RECORDING_ARTISTS'].split(', ')
        recording_title = row['RECORDING_TITLE']
        artists['name_lowered'] = artists['name'].apply(lambda x: str(x).lower())

        graph_recording = recordings[recordings['assetID'] == recording_id].index
        if graph_recording.empty:
            recordings.loc[len(recordings.index)] = [recording_title, recording_id, 0]
        graph_recording = recordings[recordings['assetID'] == recording_id].index
        if recording_isrc == recording_isrc:
            graph_isrc = isrcs[isrcs['isrc'] == recording_isrc].index
            if graph_isrc.empty:
                isrcs.loc[len(isrcs.index)] = recording_isrc
            graph_isrc = isrcs[isrcs['isrc'] == recording_isrc].index
            has_isrc.loc[len(has_isrc.index)] = [graph_recording[0], graph_isrc[0]]
        for recording_artist in recording_artists:
            graph_artist = artists[artists['name_lowered'] == str(recording_artist).lower()].index
            if graph_artist.empty:
                artists.loc[len(artists.index)] = [recording_artist, str(recording_artist).lower()]
            graph_artist = artists[artists['name_lowered'] == str(recording_artist).lower()].index
            performed.loc[len(performed.index)] = [graph_artist[0], graph_recording[0]]

        # check composition
        composition_id = row['COMPOSITION_ID']
        composition_artists = row['COMPOSITION_WRITERS'].split('/ ')
        composition_title = row['COMPOSITION_TITLE']
        client_name = row['DISPLAY_NAME']

        graph_composition = compositions[compositions['share_asset_id'] == composition_id].index
        if graph_composition.empty:
            compositions.loc[len(compositions.index)] = [composition_id, composition_title]
        graph_composition = compositions[compositions['share_asset_id'] == composition_id].index
        for composition_artist in composition_artists:
            graph_artist = artists[artists['name_lowered'] == str(composition_artist).lower()].index
            if graph_artist.empty:
                artists.loc[len(artists.index)] = [composition_artist, str(composition_artist).lower()]
            graph_artist = artists[artists['name_lowered'] == str(composition_artist).lower()].index
            wrote.loc[(len(wrote.index))] = [graph_artist[0], graph_composition[0]]
        graph_client = clients[clients['client_name'] == client_name].index
        if graph_client.empty:
            clients.loc[len(clients.index)] = client_name
        graph_client = clients[clients['client_name'] == client_name].index
        owns.loc[len(owns.index)] = [graph_client[0], graph_composition[0], 0, 0, 0]

        embedded.loc[len(embedded.index)] = [graph_composition[0], graph_recording[0]]

    recordings['recordings_index'] = recordings.index
    compositions['compositions_index'] = compositions.index
    clients['clients_index'] = clients.index
    artists['artists_index'] = artists.index
    iswcs['iswcs_index'] = iswcs.index
    isrcs['isrcs_index'] = isrcs.index

    proposed_matches = proposed_matches.merge(recordings[['assetID', 'recordings_index']],
                                              left_on='RECORDING_ID', right_on='assetID', how='inner')
    proposed_matches = proposed_matches.merge(compositions[['share_asset_id', 'compositions_index']],
                                              left_on='COMPOSITION_ID', right_on='share_asset_id', how='inner')
    proposed_matches[['compositions_index', 'recordings_index']].to_csv((osp.join(raw_data, 'proposed_matches.csv')),
                                                                        index=False)

    savez_compressed(osp.join(raw_data, 'test', 'iswcs.npz'), iswcs['iswc'].values)
    savez_compressed(osp.join(raw_data, 'test', 'isrcs.npz'), isrcs['isrc'].values)
    savez_compressed(osp.join(raw_data, 'test', 'artists.npz'), artists['name'].values)
    savez_compressed(osp.join(raw_data, 'test', 'clients.npz'), clients['client_name'].values)
    savez_compressed(osp.join(raw_data, 'test', 'recordings.npz'), recordings['recording_title'].values)
    savez_compressed(osp.join(raw_data, 'test', 'compositions.npz'), compositions['composition_title'].values)

    # embedded relation
    embedded = embedded.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    embedded = embedded.merge(recordings[['assetID', 'recordings_index']], on='assetID')
    embedded = embedded.drop(columns=['share_asset_id', 'assetID'])

    # has_isrc relation
    has_isrc = has_isrc.merge(recordings[['assetID', 'recordings_index']], on='assetID')
    has_isrc = has_isrc.merge(isrcs[['isrc', 'isrcs_index']], on='isrc')
    has_isrc = has_isrc.drop(columns=['assetID', 'isrc'])

    # has_iswc relation
    has_iswc = has_iswc.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    has_iswc = has_iswc.merge(iswcs[['iswc', 'iswcs_index']], on='iswc')
    has_iswc = has_iswc.drop(columns=['share_asset_id', 'iswc'])

    # owns relation
    owns = owns.merge(clients[['client_name', 'clients_index']], on='client_name')
    owns = owns.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    owns = owns.drop(columns=['share_asset_id', 'client_name', 'custom_id', 'share', 'policy'])

    # performed relation
    performed = performed.merge(artists[['name', 'artists_index']], on='name')
    performed = performed.merge(recordings[['assetID', 'recordings_index']], on='assetID')
    performed = performed.drop(columns=['name', 'assetID'])

    # wrote relation
    wrote = wrote.merge(artists[['name', 'artists_index']], on='name')
    wrote = wrote.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    wrote = wrote.drop(columns=['share_asset_id', 'name'])

    data['recording'].x = torch.from_numpy(recordings['recordings_index'].values)
    data['composition'].x = torch.from_numpy(compositions['compositions_index'].values)
    data['artist'].x = torch.from_numpy(artists['artists_index'].values)
    data['client'].x = torch.from_numpy(clients['clients_index'].values)
    data['isrc'].x = torch.from_numpy(isrcs['isrcs_index'].values)
    data['iswc'].x = torch.from_numpy(iswcs['iswcs_index'].values)

    print(f'Processing composition_embedded_recording relation...')
    data[('composition', 'embedded', 'recording')].edge_index = torch.from_numpy(embedded.values).t().contiguous()

    print(f'Processing recording_has_isrc_isrc relation...')
    data[('recording', 'has_isrc', 'isrc')].edge_index = torch.from_numpy(has_isrc.values).t().contiguous()

    print(f'Processing composition_has_iswc_iswc relation...')
    data[('composition', 'has_iswc', 'iswc')].edge_index = torch.from_numpy(has_iswc.values).t().contiguous()

    print(f'Processing client_owns_composition relation...')
    data[('client', 'owns', 'composition')].edge_index = torch.from_numpy(owns.values).t().contiguous()

    print(f'Processing artist_performed_recording relation...')
    data[('artist', 'performed', 'recording')].edge_index = torch.from_numpy(performed.values).t().contiguous()

    print(f'Processing artist_wrote_composition relation...')
    data[('artist', 'wrote', 'composition')].edge_index = torch.from_numpy(wrote.values).t().contiguous()

    transform = T.ToUndirected()
    data = transform(data)

    hetero_data_path = osp.join(raw_data, 'test', 'data.pt')
    torch.save(data, hetero_data_path)
    print(f'Hetero dataset was saved to {hetero_data_path}')


def create_raw_graph_data_from_raw(sample_size: int, raw_data: str, raw_graph_data: str):
    print('Loading raw data...')
    recordings = pd.read_csv(osp.join(raw_data, 'recording.csv'))
    recordings = recordings.dropna(subset=['recording_title']).reset_index(drop=True)

    compositions = pd.read_csv(osp.join(raw_data, 'composition.csv'))
    compositions = compositions.dropna(subset=['composition_title']).reset_index(drop=True)
    clients = pd.read_csv(osp.join(raw_data, 'client.csv'))
    artists = pd.read_csv(osp.join(raw_data, 'artist.csv'))
    iswcs = pd.read_csv(osp.join(raw_data, 'iswc.csv'))
    isrcs = pd.read_csv(osp.join(raw_data, 'isrc.csv'))

    embedded = pd.read_csv(osp.join(raw_data, 'embedded.csv'))
    has_isrc = pd.read_csv(osp.join(raw_data, 'has_isrc.csv'))
    has_iswc = pd.read_csv(osp.join(raw_data, 'has_iswc.csv'))
    owns = pd.read_csv(osp.join(raw_data, 'owns.csv'))
    performed = pd.read_csv(osp.join(raw_data, 'performed.csv'))
    wrote = pd.read_csv(osp.join(raw_data, 'wrote.csv'))
    # neg_embedded = pd.read_csv(osp.join(raw_data, 'neg_embedded.csv'))
    proposed_matches = pd.read_csv(osp.join(raw_data, 'results_april_all_clients.csv'))[:150]

    test(raw_data, recordings, compositions, artists, clients, iswcs, isrcs,
         embedded, has_isrc, has_iswc, owns, performed, wrote, proposed_matches)

    import sys
    sys.exit(-1)

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
    recordings_path = osp.join(raw_graph_data, 'node-feat', 'recording_features.npz')
    recordings = np.load(recordings_path)['arr_0']
    data['recording'].x = torch.from_numpy(recordings)

    print(f'Processing composition nodes...')
    compositions_path = osp.join(raw_graph_data, 'node-feat', 'composition_features.npz')
    compositions = np.load(compositions_path)['arr_0']
    data['composition'].x = torch.from_numpy(compositions)

    print(f'Processing artist nodes...')
    artists_path = osp.join(raw_graph_data, 'node-feat', 'artist_features.npz')
    artists = np.load(artists_path)['arr_0']
    data['artist'].x = torch.from_numpy(artists)

    print(f'Processing isrc nodes...')
    isrcs_path = osp.join(raw_graph_data, 'node-feat', 'isrcs_features.npz')
    isrcs = np.load(isrcs_path)['arr_0']
    data['isrc'].x = torch.from_numpy(isrcs)

    print(f'Processing iswc nodes...')
    iswcs_path = osp.join(raw_graph_data, 'node-feat', 'iswcs_features.npz')
    iswcs = np.load(iswcs_path)['arr_0']
    data['iswc'].x = torch.from_numpy(iswcs)

    print(f'Processing client nodes...')
    clients_path = osp.join(raw_graph_data, 'node-feat', 'clients_features.npz')
    clients = np.load(clients_path)['arr_0']
    data['client'].x = torch.from_numpy(clients)


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
