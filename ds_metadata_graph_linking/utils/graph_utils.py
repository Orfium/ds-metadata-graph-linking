import pandas as pd


def sample_graph(compositions: pd.DataFrame, recordings: pd.DataFrame, clients: pd.DataFrame,
                 iswcs: pd.DataFrame, isrcs: pd.DataFrame, embedded: pd.DataFrame, performed: pd.DataFrame,
                 wrote: pd.DataFrame, has_isrc: pd.DataFrame,
                 has_iswc: pd.DataFrame, owns: pd.DataFrame, compositions_to_sample: int):
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
    compositions_sample.to_csv('data/base_dataset/processed_dataset/compositions_sample.csv', index=False)
    has_iswc_sample.to_csv('data/base_dataset/processed_dataset/has_iswc_sample.csv', index=False)
    iswcs_sample.to_csv('data/base_dataset/processed_dataset/iswcs_sample.csv', index=False)
    embedded_sample.to_csv('data/base_dataset/processed_dataset/embedded_sample.csv', index=False)
    recordings_sample.to_csv('data/base_dataset/processed_dataset/recordings_sample.csv', index=False)
    has_isrc_sample.to_csv('data/base_dataset/processed_dataset/has_isrc_sample.csv', index=False)
    isrcs_sample.to_csv('data/base_dataset/processed_dataset/isrcs_sample.csv', index=False)
    owns_sample.to_csv('data/base_dataset/processed_dataset/owns_sample.csv', index=False)
    client_sample.to_csv('data/base_dataset/processed_dataset/client_sample.csv', index=False)
    performed_sample.to_csv('data/base_dataset/processed_dataset/performed_sample.csv', index=False)
    wrote_sample.to_csv('data/base_dataset/processed_dataset/wrote_sample.csv', index=False)
    artists_sample.to_csv('data/base_dataset/processed_dataset/artists_sample.csv', index=False)


def generate_graph_statistics(compositions: pd.DataFrame, recordings: pd.DataFrame, clients: pd.DataFrame,
                              iswcs: pd.DataFrame, isrcs: pd.DataFrame, artists: pd.DataFrame,
                              embedded: pd.DataFrame, performed: pd.DataFrame, wrote: pd.DataFrame,
                              has_isrc: pd.DataFrame, has_iswc: pd.DataFrame, owns: pd.DataFrame):

    recordings_nodes = recordings.shape[0]
    compositions_nodes = compositions.shape[0]
    clients_nodes = clients.shape[0]
    iswcs_nodes = iswcs.shape[0]
    isrcs_nodes = isrcs.shape[0]
    artists_nodes = artists.shape[0]

    embedded_edges = embedded.shape[0]
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
    print(f'has_isrc edges: {has_isrc_edges}')
    print(f'has_iswc edges: {has_iswc_edges}')
    print(f'owns edges: {owns_edges}')
    print(f'performed edges: {performed_edges}')
    print(f'wrote edges: {wrote_edges}')


def generate_indexes_for_relations(compositions: pd.DataFrame, recordings: pd.DataFrame, clients: pd.DataFrame,
                                   iswcs: pd.DataFrame, isrcs: pd.DataFrame, artists: pd.DataFrame,
                                   embedded: pd.DataFrame, performed: pd.DataFrame, wrote: pd.DataFrame,
                                   has_isrc: pd.DataFrame, has_iswc: pd.DataFrame, owns: pd.DataFrame):
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
    embedded.to_csv('data/base_dataset/processed_dataset/embedded_sample.csv', index=False)

    # has_isrc relation
    has_isrc = has_isrc.merge(recordings[['assetID', 'recordings_index']], on='assetID')
    has_isrc = has_isrc.merge(isrcs[['isrc', 'isrcs_index']], on='isrc')
    has_isrc = has_isrc.drop(columns=['assetID', 'isrc'])
    has_isrc.to_csv('data/base_dataset/processed_dataset/has_isrc_sample.csv', index=False)

    # has_iswc relation
    has_iswc = has_iswc.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    has_iswc = has_iswc.merge(iswcs[['iswc', 'iswcs_index']], on='iswc')
    has_iswc = has_iswc.drop(columns=['share_asset_id', 'iswc'])
    has_iswc.to_csv('data/base_dataset/processed_dataset/has_iswc_sample.csv', index=False)

    # owns relation
    owns = owns.merge(clients[['client_name', 'clients_index']], on='client_name')
    owns = owns.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    owns = owns.drop(columns=['share_asset_id', 'client_name', 'custom_id', 'share', 'policy'])
    owns.to_csv('data/base_dataset/processed_dataset/owns_sample.csv', index=False)

    # performed relation
    performed = performed.merge(artists[['name', 'artists_index']], on='name')
    performed = performed.merge(recordings[['assetID', 'recordings_index']], on='assetID')
    performed = performed.drop(columns=['name', 'assetID'])
    performed.to_csv('data/base_dataset/processed_dataset/performed_sample.csv', index=False)

    # wrote relation
    wrote = wrote.merge(artists[['name', 'artists_index']], on='name')
    wrote = wrote.merge(compositions[['share_asset_id', 'compositions_index']], on='share_asset_id')
    wrote = wrote.drop(columns=['share_asset_id', 'name'])
    wrote.to_csv('data/base_dataset/processed_dataset/wrote_sample.csv', index=False)
