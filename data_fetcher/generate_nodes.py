import warnings
import pandas as pd

from data_fetcher.headers import write_headers
from data_fetcher.idset import IdSet
from data_fetcher.preprocess import preprocess_artists, preprocess_writers
from data_fetcher.export import export_csv, NodePath, RelPath


def extract_recording_nodes(recordings: pd.DataFrame, raw_data_path: str):
    recordings = recordings[["ASSET_TITLE", "ASSET_ID", "VIEW_ID"]].copy()
    recordings.loc[:, 'ASSET_TITLE'] = recordings['ASSET_TITLE'].str.replace("\n", " ",
                                                                             regex=False).str.strip().str.title()
    export_csv(recordings, NodePath.Recording, raw_data_path)


def extract_isrcs(recordings: pd.DataFrame, raw_data_path: str):
    isrc_df = recordings[["ASSET_ID", "ISRC"]].dropna().drop_duplicates()
    export_csv(isrc_df, RelPath.HAS_ISRC, raw_data_path)
    isrc_df = isrc_df[["ISRC"]].drop_duplicates()
    export_csv(isrc_df, NodePath.ISRC, raw_data_path)


def extract_artist_rels(recordings: pd.DataFrame, artist_set: IdSet, raw_data_path: str):
    recordings = preprocess_artists(recordings)
    artist_of = set()
    for row in recordings.itertuples(index=False):
        row_artists = map(artist_set.add, row.ASSET_ARTIST.split(', '))
        for artist in row_artists:
            artist_of.add((artist, row.ASSET_ID))

    artist_of_rels = pd.DataFrame(artist_of)
    export_csv(artist_of_rels, RelPath.PERFORMED, raw_data_path)


def process_recordings(assets: pd.DataFrame, artists: IdSet, raw_data_path: str):
    recordings = assets[["ASSET_ID", "ASSET_TITLE", "ISRC", "ASSET_ARTIST", "VIEW_ID"]].drop_duplicates(
        subset="VIEW_ID")
    extract_recording_nodes(recordings, raw_data_path)
    extract_isrcs(recordings, raw_data_path)
    extract_artist_rels(recordings, artists, raw_data_path)


def extract_r2c(compositions: pd.DataFrame, raw_data_path: str):
    r2c = compositions[["SHARE_ASSET_ID", "ASSET_ID"]].drop_duplicates()
    export_csv(r2c, RelPath.EMBEDDED, raw_data_path)


def extract_composition_nodes(compositions: pd.DataFrame, raw_data_path: str):
    compositions = compositions[["SHARE_ASSET_ID", "ASSET_SHARE_TITLE"]].copy()
    compositions.loc[:, "ASSET_SHARE_TITLE"] = compositions["ASSET_SHARE_TITLE"].str.replace("\n", "",
                                                                                             regex=False).str.upper()
    export_csv(compositions, NodePath.Composition, raw_data_path)


def extract_iswcs(compositions: pd.DataFrame, raw_data_path: str):
    compositions = compositions[["SHARE_ASSET_ID", "ISWC"]].dropna()
    export_csv(compositions, RelPath.HAS_ISWC, raw_data_path)
    iswc_df = compositions[["ISWC"]].drop_duplicates()
    export_csv(iswc_df, NodePath.ISWC, raw_data_path)


def extract_hfa_codes(compositions: pd.DataFrame, raw_data_path: str):
    compositions = compositions[["SHARE_ASSET_ID", "HFA_CODE"]].dropna()
    export_csv(compositions, RelPath.HAS_HFA_CODE, raw_data_path)
    hfa_df = compositions[["HFA_CODE"]].drop_duplicates()
    export_csv(hfa_df, NodePath.HFA_CODE, raw_data_path)


def extract_writers(compositions: pd.DataFrame, artist_set: IdSet, raw_data_path: str):
    compositions = preprocess_writers(compositions)
    writer_of = set()
    for row in compositions.itertuples(index=False):
        writers = map(artist_set.add, row.ASSET_WRITERS.split('/'))
        for writer in writers:
            writer_of.add((writer, row.SHARE_ASSET_ID))

    writer_of_rels = pd.DataFrame(writer_of)
    export_csv(writer_of_rels, RelPath.WROTE, raw_data_path)


def process_compositions(assets: pd.DataFrame, artists: IdSet, raw_data_path: str):
    compositions = assets[["ASSET_ID", "SHARE_ASSET_ID", "ISWC", "ASSET_SHARE_TITLE", "ASSET_WRITERS", "HFA_CODE"]]
    extract_r2c(compositions, raw_data_path)
    compositions = compositions.drop_duplicates("SHARE_ASSET_ID")
    extract_composition_nodes(compositions, raw_data_path)
    extract_iswcs(compositions, raw_data_path)
    extract_hfa_codes(compositions, raw_data_path)
    extract_writers(compositions, artists, raw_data_path)


def extract_artist_nodes(artist_set: IdSet, raw_data_path: str):
    print('Processing Artists')
    artists = pd.DataFrame({"name": list(artist_set.id)})
    export_csv(artists, NodePath.Artist, raw_data_path)


def process_assets(assets: pd.DataFrame, artists: IdSet, raw_data_path: str):
    print('Processing Recordings')
    process_recordings(assets, artists, raw_data_path)
    print('Processing Compositions')
    process_compositions(assets, artists, raw_data_path)


def process_shares(shares: pd.DataFrame, raw_data_path: str):
    print('Processing shares')
    shares["US"] = shares["OWNERSHIP_PROVIDED"].str.extract(r'([\d.]+)% Owned US')
    shares["Everywhere"] = shares["OWNERSHIP_PROVIDED"].str.extract(r'([\d.]+)% Owned Everywhere')
    shares["Elsewhere"] = shares["OWNERSHIP_PROVIDED"].str.extract(r'([\d.]+)% Owned Elsewhere')
    shares["Ownership"] = shares["US"]
    shares.loc[shares["Ownership"].isnull(), "Ownership"] = shares.loc[shares["Ownership"].isnull(), "Everywhere"]
    shares.loc[shares["Ownership"].isnull(), "Ownership"] = shares.loc[shares["Ownership"].isnull(), "Elsewhere"]
    shares["Ownership"] = shares["Ownership"].fillna("0")
    shares = shares[["CLIENT", "SHARE_ASSET_ID", "CUSTOM_ID", "Ownership", "POLICY"]]
    export_csv(shares, RelPath.OWNS, raw_data_path)
    clients = shares[["CLIENT"]].drop_duplicates()
    export_csv(clients, NodePath.Client, raw_data_path)


def generate_nodes(asset_full_path, asset_share_path, raw_data_path):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    write_headers(raw_data_path)
    assets = pd.read_csv(asset_full_path)
    print('Read asset full')
    artists = IdSet("ar")
    process_assets(assets, artists, raw_data_path)
    extract_artist_nodes(artists, raw_data_path)
    print('Asset full processed')
    shares = pd.read_csv(asset_share_path)
    print('Read asset share')
    process_shares(shares, raw_data_path)
    print('Asset share processed')
