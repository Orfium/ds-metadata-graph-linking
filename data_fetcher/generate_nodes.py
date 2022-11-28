import pandas as pd
from dotenv import load_dotenv

from data_fetcher.headers import write_headers
from data_fetcher.idset import IdSet
from data_fetcher.preprocess import preprocess_artists, preprocess_writers
from data_fetcher.export import export_csv, NodePath, RelPath


def extract_recording_nodes(recordings: pd.DataFrame):
    recordings = recordings[["ASSET_TITLE", "ASSET_ID", "VIEW_ID"]].copy()
    recordings.loc[:, 'ASSET_TITLE'] = recordings['ASSET_TITLE'].str.replace("\n", " ",
                                                                             regex=False).str.strip().str.title()
    export_csv(recordings, NodePath.Recording)


def extract_isrcs(recordings: pd.DataFrame):
    isrc_df = recordings[["ASSET_ID", "ISRC"]].dropna().drop_duplicates()
    export_csv(isrc_df, RelPath.HAS_ISRC)
    isrc_df = isrc_df[["ISRC"]].drop_duplicates()
    export_csv(isrc_df, NodePath.ISRC)


def extract_artist_rels(recordings: pd.DataFrame, artist_set: IdSet):
    recordings = preprocess_artists(recordings)
    artist_of = set()
    for row in recordings.itertuples(index=False):
        row_artists = map(artist_set.add, row.ASSET_ARTIST.split(', '))
        for artist in row_artists:
            artist_of.add((artist, row.ASSET_ID))

    artist_of_rels = pd.DataFrame(artist_of)
    export_csv(artist_of_rels, RelPath.PERFORMED)


def process_recordings(assets: pd.DataFrame, artists: IdSet):
    recordings = assets[["ASSET_ID", "ASSET_TITLE", "ISRC", "ASSET_ARTIST", "VIEW_ID"]].drop_duplicates(subset="VIEW_ID")
    extract_recording_nodes(recordings)
    extract_isrcs(recordings)
    extract_artist_rels(recordings, artists)


def extract_r2c(compositions: pd.DataFrame):
    r2c = compositions[["SHARE_ASSET_ID", "ASSET_ID"]].drop_duplicates()
    export_csv(r2c, RelPath.EMBEDDED)


def extract_composition_nodes(compositions: pd.DataFrame):
    compositions = compositions[["SHARE_ASSET_ID", "ASSET_SHARE_TITLE"]].copy()
    compositions.loc[:, "ASSET_SHARE_TITLE"] = compositions["ASSET_SHARE_TITLE"].str.replace("\n", "",
                                                                                             regex=False).str.upper()
    export_csv(compositions, NodePath.Composition)


def extract_iswcs(compositions: pd.DataFrame):
    compositions = compositions[["SHARE_ASSET_ID", "ISWC"]].dropna()
    export_csv(compositions, RelPath.HAS_ISWC)
    iswc_df = compositions[["ISWC"]].drop_duplicates()
    export_csv(iswc_df, NodePath.ISWC)


def extract_hfa_codes(compositions: pd.DataFrame):
    compositions = compositions[["SHARE_ASSET_ID", "HFA_CODE"]].dropna()
    export_csv(compositions, RelPath.HAS_HFA_CODE)
    hfa_df = compositions[["HFA_CODE"]].drop_duplicates()
    export_csv(hfa_df, NodePath.HFA_CODE)


def extract_writers(compositions: pd.DataFrame, artist_set: IdSet):
    compositions = preprocess_writers(compositions)
    writer_of = set()
    for row in compositions.itertuples(index=False):
        writers = map(artist_set.add, row.ASSET_WRITERS.split('/'))
        for writer in writers:
            writer_of.add((writer, row.SHARE_ASSET_ID))

    writer_of_rels = pd.DataFrame(writer_of)
    export_csv(writer_of_rels, RelPath.WROTE)


def process_compositions(assets: pd.DataFrame, artists: IdSet):
    compositions = assets[["ASSET_ID", "SHARE_ASSET_ID", "ISWC", "ASSET_SHARE_TITLE", "ASSET_WRITERS", "HFA_CODE"]]
    extract_r2c(compositions)
    compositions = compositions.drop_duplicates("SHARE_ASSET_ID")
    extract_composition_nodes(compositions)
    extract_iswcs(compositions)
    extract_hfa_codes(compositions)
    extract_writers(compositions, artists)


def extract_artist_nodes(artist_set: IdSet):
    artists = pd.DataFrame({"name": list(artist_set.id)})
    export_csv(artists, NodePath.Artist)


def process_assets(assets: pd.DataFrame, artists: IdSet):
    process_recordings(assets, artists)
    process_compositions(assets, artists)


def process_shares(shares: pd.DataFrame):
    # shares = shares.copy()
    shares["US"] = shares["OWNERSHIP_PROVIDED"].str.extract(r'([\d.]+)% Owned US')
    shares["Everywhere"] = shares["OWNERSHIP_PROVIDED"].str.extract(r'([\d.]+)% Owned Everywhere')
    shares["Elsewhere"] = shares["OWNERSHIP_PROVIDED"].str.extract(r'([\d.]+)% Owned Elsewhere')
    shares["Ownership"] = shares["US"]
    shares.loc[shares["Ownership"].isnull(), "Ownership"] = shares.loc[shares["Ownership"].isnull(), "Everywhere"]
    shares.loc[shares["Ownership"].isnull(), "Ownership"] = shares.loc[shares["Ownership"].isnull(), "Elsewhere"]
    shares["Ownership"] = shares["Ownership"].fillna("0")
    shares = shares[["CLIENT", "SHARE_ASSET_ID", "CUSTOM_ID", "Ownership", "POLICY"]]
    export_csv(shares, RelPath.OWNS)
    clients = shares[["CLIENT"]].drop_duplicates()
    export_csv(clients, NodePath.Client)


def main():
    load_dotenv()
    write_headers()
    assets = pd.read_csv('asset_full_latest.csv')
    print('Read asset full')
    artists = IdSet("ar")
    process_assets(assets, artists)
    extract_artist_nodes(artists)
    print('Asset full processed')
    shares = pd.read_csv('asset_share_latest.csv')
    print('Read asset share')
    process_shares(shares)
    print('Asset share processed')


if __name__ == '__main__':
    main()
