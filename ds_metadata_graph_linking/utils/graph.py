import os
import pandas as pd


def negative_sampling(neg_embedded, embedded_sample, embedded_to_sample):
    columns = ['assetID', 'share_asset_id']
    neg_embedded = neg_embedded.drop_duplicates(subset=columns)
    
    if embedded_to_sample > neg_embedded.shape[0]:
        print(f'Can not sample {embedded_to_sample} negatives, adjusting it to {neg_embedded.shape[0]}')
        embedded_to_sample = neg_embedded.shape[0]

    neg_embedded_sample = embedded_sample.merge(neg_embedded, on='share_asset_id', how='inner')
    neg_embedded_sample = neg_embedded_sample[['share_asset_id', 'assetID_y', 'neg_embedded_sample_index']]
    neg_embedded_sample = neg_embedded_sample.drop_duplicates()
    neg_embedded_sample = neg_embedded_sample.rename(columns={"assetID_y": "assetID"})  # rename after merging

    # in most cases this sample is not enough, so we need to sample more from the remaining initial pool
    if neg_embedded_sample.shape[0] < embedded_to_sample:
        embedded_to_sample_left = embedded_to_sample - neg_embedded_sample.shape[0]
        sample_criterion = neg_embedded.index.isin(neg_embedded_sample['neg_embedded_sample_index'].values)
        neg_embedded_to_sample_from = neg_embedded[~sample_criterion]
        neg_embedded_sample_extra = neg_embedded_to_sample_from.sample(embedded_to_sample_left)
        neg_embedded_sample = pd.concat([neg_embedded_sample[columns], neg_embedded_sample_extra[columns]])
        neg_embedded_sample = neg_embedded_sample.reset_index(drop=True)

    return neg_embedded_sample


def link_sample_graph(compositions: pd.DataFrame, recordings: pd.DataFrame, clients: pd.DataFrame,
                      iswcs: pd.DataFrame, isrcs: pd.DataFrame, embedded: pd.DataFrame, neg_embedded: pd.DataFrame,
                      performed: pd.DataFrame, wrote: pd.DataFrame, has_isrc: pd.DataFrame,
                      has_iswc: pd.DataFrame, owns: pd.DataFrame, embedded_to_sample: int, dataset_to_save: str):
    neg_embedded['neg_embedded_sample_index'] = neg_embedded.index  # index the negatives

    embedded_sample = embedded.sample(embedded_to_sample).drop_duplicates()
    neg_embedded_sample = negative_sampling(neg_embedded, embedded_sample, embedded_to_sample)

    pos_neg_embedded_sample = pd.concat([
        embedded_sample[['assetID', 'share_asset_id']],
        neg_embedded_sample[['assetID', 'share_asset_id']]
    ], keys=['assetID', 'share_asset_id']).drop_duplicates().reset_index(drop=True)

    # sample tbe nodes based on the concat pos and neg edges
    compositions_sample = compositions.merge(pos_neg_embedded_sample,
                                             on='share_asset_id',
                                             how='inner')[compositions.columns].drop_duplicates()

    has_iswc_sample = has_iswc.merge(compositions_sample,
                                     on='share_asset_id',
                                     how='inner')[has_iswc.columns].drop_duplicates()
    owns_sample = owns.merge(compositions_sample, on='share_asset_id', how='inner')[owns.columns].drop_duplicates()
    client_sample = clients.merge(owns_sample, on='client_name', how='inner')[clients.columns].drop_duplicates()
    wrote_sample = wrote.merge(compositions_sample, on='share_asset_id', how='inner')[wrote.columns].drop_duplicates()

    recordings_sample = recordings.merge(pos_neg_embedded_sample,
                                         on='assetID',
                                         how='inner')[recordings.columns].drop_duplicates()

    has_isrc_sample = has_isrc.merge(recordings_sample, on='assetID', how='inner')[has_isrc.columns].drop_duplicates()
    performed_sample = performed.merge(recordings_sample,
                                       on='assetID',
                                       how='inner')[performed.columns].drop_duplicates()
    artists_sample = pd.concat([performed_sample[['name']],
                                wrote_sample[['name']]]).drop_duplicates().reset_index(drop=True)

    isrcs_sample = isrcs.merge(has_isrc_sample, on='isrc', how='inner')[isrcs.columns].drop_duplicates()
    iswcs_sample = iswcs.merge(has_iswc_sample, on='iswc', how='inner')[iswcs.columns].drop_duplicates()

    compositions_sample.to_csv(os.path.join(dataset_to_save, 'node-feat/compositions_sample.csv'), index=False)
    has_iswc_sample.to_csv(os.path.join(dataset_to_save, 'relations/has_iswc_sample.csv'), index=False)
    iswcs_sample.to_csv(os.path.join(dataset_to_save, 'node-feat/iswcs_sample.csv'), index=False)
    embedded_sample.to_csv(os.path.join(dataset_to_save, 'relations/embedded_sample.csv'), index=False)
    neg_embedded_sample.to_csv(os.path.join(dataset_to_save, 'relations/neg_embedded_sample.csv'), index=False)
    recordings_sample.to_csv(os.path.join(dataset_to_save, 'node-feat/recordings_sample.csv'), index=False)
    has_isrc_sample.to_csv(os.path.join(dataset_to_save, 'relations/has_isrc_sample.csv'), index=False)
    isrcs_sample.to_csv(os.path.join(dataset_to_save, 'node-feat/isrcs_sample.csv'), index=False)
    owns_sample.to_csv(os.path.join(dataset_to_save, 'relations/owns_sample.csv'), index=False)
    client_sample.to_csv(os.path.join(dataset_to_save, 'node-feat/client_sample.csv'), index=False)
    performed_sample.to_csv(os.path.join(dataset_to_save, 'relations/performed_sample.csv'), index=False)
    wrote_sample.to_csv(os.path.join(dataset_to_save, 'relations/wrote_sample.csv'), index=False)
    artists_sample.to_csv(os.path.join(dataset_to_save, 'node-feat/artist_sample.csv'), index=False)


def sample_graph(compositions: pd.DataFrame, recordings: pd.DataFrame, clients: pd.DataFrame,
                 iswcs: pd.DataFrame, isrcs: pd.DataFrame, embedded: pd.DataFrame, performed: pd.DataFrame,
                 wrote: pd.DataFrame, has_isrc: pd.DataFrame,
                 has_iswc: pd.DataFrame, owns: pd.DataFrame, compositions_to_sample: int, dataset_to_save: str):
    # composition and relations sample
    compositions_sample = compositions.sample(compositions_to_sample)
    has_iswc_sample = has_iswc.merge(compositions_sample, on='share_asset_id', how='inner')[
        has_iswc.columns].drop_duplicates()
    iswcs_sample = iswcs.merge(has_iswc_sample, on='iswc', how='inner')[iswcs.columns].drop_duplicates()
    embedded_sample = embedded.merge(compositions_sample, on='share_asset_id', how='inner')[
        embedded.columns].drop_duplicates()

    # recordings and relations sample
    recordings_sample = recordings.merge(embedded_sample, on='assetID', how='inner')[
        recordings.columns].drop_duplicates()
    has_isrc_sample = has_isrc.merge(recordings_sample, on='assetID', how='inner')[has_isrc.columns].drop_duplicates()
    isrcs_sample = isrcs.merge(has_isrc_sample, on='isrc', how='inner')[isrcs.columns].drop_duplicates()

    # clients and relations sample
    owns_sample = owns.merge(compositions_sample, on='share_asset_id', how='inner')[owns.columns].drop_duplicates()
    client_sample = clients.merge(owns_sample, on='client_name', how='inner')[clients.columns].drop_duplicates()

    # artists and relations sample
    performed_sample = performed.merge(recordings_sample, on='assetID', how='inner')[
        performed.columns].drop_duplicates()
    wrote_sample = wrote.merge(compositions_sample, on='share_asset_id', how='inner')[wrote.columns].drop_duplicates()
    to_find_artists = pd.concat([performed_sample[['name']], wrote_sample[['name']]]).drop_duplicates().reset_index(
        drop=True)
    artists_sample = to_find_artists

    # save to csv
    compositions_sample.to_csv(os.path.join(dataset_to_save, 'nodes/compositions_sample.csv'), index=False)
    has_iswc_sample.to_csv(os.path.join(dataset_to_save, 'relations/has_iswc_sample.csv'), index=False)
    iswcs_sample.to_csv(os.path.join(dataset_to_save, 'nodes/iswcs_sample.csv'), index=False)
    embedded_sample.to_csv(os.path.join(dataset_to_save, 'relations/embedded_sample.csv'), index=False)
    recordings_sample.to_csv(os.path.join(dataset_to_save, 'nodes/recordings_sample.csv'), index=False)
    has_isrc_sample.to_csv(os.path.join(dataset_to_save, 'relations/has_isrc_sample.csv'), index=False)
    isrcs_sample.to_csv(os.path.join(dataset_to_save, 'nodes/isrcs_sample.csv'), index=False)
    owns_sample.to_csv(os.path.join(dataset_to_save, 'relations/owns_sample.csv'), index=False)
    client_sample.to_csv(os.path.join(dataset_to_save, 'nodes/client.csv'), index=False)
    performed_sample.to_csv(os.path.join(dataset_to_save, 'relations/performed_sample.csv'), index=False)
    wrote_sample.to_csv(os.path.join(dataset_to_save, 'relations/wrote_sample.csv'), index=False)
    artists_sample.to_csv(os.path.join(dataset_to_save, 'nodes/artist.csv'), index=False)


def generate_graph_statistics(compositions: pd.DataFrame, recordings: pd.DataFrame, clients: pd.DataFrame,
                              iswcs: pd.DataFrame, isrcs: pd.DataFrame, artists: pd.DataFrame,
                              embedded: pd.DataFrame, neg_embedded: pd.DataFrame,
                              performed: pd.DataFrame, wrote: pd.DataFrame, has_isrc: pd.DataFrame,
                              has_iswc: pd.DataFrame, owns: pd.DataFrame):
    recordings_nodes = recordings.shape[0]
    compositions_nodes = compositions.shape[0]
    clients_nodes = clients.shape[0]
    iswcs_nodes = iswcs.shape[0]
    isrcs_nodes = isrcs.shape[0]
    artists_nodes = artists.shape[0]

    embedded_edges = embedded.shape[0]
    neg_embedded_edges = neg_embedded.shape[0]
    has_isrc_edges = has_isrc.shape[0]
    has_iswc_edges = has_iswc.shape[0]
    owns_edges = owns.shape[0]
    performed_edges = performed.shape[0]
    wrote_edges = wrote.shape[0]

    total_number_of_nodes = recordings.shape[0] + compositions.shape[0] + clients.shape[0] + iswcs.shape[0] + \
                            isrcs.shape[0]
    total_number_of_edges = embedded.shape[0] + has_isrc.shape[0] + has_iswc.shape[0] + owns.shape[0] + performed.shape[
        0] + wrote.shape[0]
    average_degree = total_number_of_edges / total_number_of_nodes

    print(f'Total number of nodes: {total_number_of_nodes}')
    print(f'Total number of edges: {total_number_of_edges}')
    print(f'Average degree in Graph: {average_degree}')
    print()
    print(f'Artist nodes: {artists_nodes}')
    print(f'Recording nodes: {recordings_nodes}')
    print(f'Composition nodes: {compositions_nodes}')
    print(f'Client nodes: {clients_nodes}')
    print(f'ISWC nodes: {iswcs_nodes}')
    print(f'ISRC nodes: {isrcs_nodes}')
    print()
    print(f'embedded edges: {embedded_edges}')
    print(f'neg_embedded edges: {neg_embedded_edges}')
    print(f'has_isrc edges: {has_isrc_edges}')
    print(f'has_iswc edges: {has_iswc_edges}')
    print(f'owns edges: {owns_edges}')
    print(f'performed edges: {performed_edges}')
    print(f'wrote edges: {wrote_edges}')


def generate_indexes_for_relations(compositions: pd.DataFrame, recordings: pd.DataFrame, clients: pd.DataFrame,
                                   iswcs: pd.DataFrame, isrcs: pd.DataFrame, artists: pd.DataFrame,
                                   embedded: pd.DataFrame, neg_embedded: pd.DataFrame,
                                   performed: pd.DataFrame, wrote: pd.DataFrame, has_isrc: pd.DataFrame,
                                   has_iswc: pd.DataFrame, owns: pd.DataFrame, dataset_to_save: str):
    # get node indexes
    recordings['recordings_index'] = recordings.index
    artists['artists_index'] = artists.index
    compositions['compositions_index'] = compositions.index
    clients['clients_index'] = clients.index
    iswcs['iswcs_index'] = iswcs.index
    isrcs['isrcs_index'] = isrcs.index

    # embedded relation
    embedded = embedded.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    embedded = embedded.merge(recordings[['assetID', 'recordings_index']], on='assetID')
    embedded = embedded.drop(columns=['share_asset_id', 'assetID'])
    embedded.to_csv(os.path.join(dataset_to_save, 'relations/embedded_sample.csv'), index=False)

    neg_embedded = neg_embedded.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    neg_embedded = neg_embedded.merge(recordings[['assetID', 'recordings_index']], on='assetID')
    neg_embedded = neg_embedded.drop(columns=['share_asset_id', 'assetID'])
    neg_embedded.to_csv(os.path.join(dataset_to_save, 'relations/neg_embedded_sample.csv'), index=False)

    # has_isrc relation
    has_isrc = has_isrc.merge(recordings[['assetID', 'recordings_index']], on='assetID')
    has_isrc = has_isrc.merge(isrcs[['isrc', 'isrcs_index']], on='isrc')
    has_isrc = has_isrc.drop(columns=['assetID', 'isrc'])
    has_isrc.to_csv(os.path.join(dataset_to_save, 'relations/has_isrc_sample.csv'), index=False)

    # has_iswc relation
    has_iswc = has_iswc.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    has_iswc = has_iswc.merge(iswcs[['iswc', 'iswcs_index']], on='iswc')
    has_iswc = has_iswc.drop(columns=['share_asset_id', 'iswc'])
    has_iswc.to_csv(os.path.join(dataset_to_save, 'relations/has_iswc_sample.csv'), index=False)

    # owns relation
    owns = owns.merge(clients[['client_name', 'clients_index']], on='client_name')
    owns = owns.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    owns = owns.drop(columns=['share_asset_id', 'client_name', 'custom_id', 'share', 'policy'])
    owns.to_csv(os.path.join(dataset_to_save, 'relations/owns_sample.csv'), index=False)

    # performed relation
    performed = performed.merge(artists[['name', 'artists_index']], on='name')
    performed = performed.merge(recordings[['assetID', 'recordings_index']], on='assetID')
    performed = performed.drop(columns=['name', 'assetID'])
    performed.to_csv(os.path.join(dataset_to_save, 'relations/performed_sample.csv'), index=False)

    # wrote relation
    wrote = wrote.merge(artists[['name', 'artists_index']], on='name')
    wrote = wrote.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    wrote = wrote.drop(columns=['share_asset_id', 'name'])
    wrote.to_csv(os.path.join(dataset_to_save, 'relations/wrote_sample.csv'), index=False)
