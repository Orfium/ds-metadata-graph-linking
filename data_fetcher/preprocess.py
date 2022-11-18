import re

import pandas as pd


def preprocess_artists(recordings: pd.DataFrame):
    recordings = recordings[["ASSET_ARTIST", "ASSET_ID"]].dropna().copy()
    recordings.loc[:, 'ASSET_ARTIST'] = recordings['ASSET_ARTIST'].str.replace("\n", " ", regex=False)
    recordings.loc[:, 'ASSET_ARTIST'] = recordings['ASSET_ARTIST'].str.replace(r"[;/\|]", ",", regex=True)
    recordings.loc[:, 'ASSET_ARTIST'] = recordings['ASSET_ARTIST'].str.replace(r',,', ',', regex=True)
    recordings.loc[:, 'ASSET_ARTIST'] = recordings['ASSET_ARTIST'].str.replace(r',([^\s])', r', \1', regex=True)
    return recordings


def preprocess_writers(compositions: pd.DataFrame):
    compositions = compositions[["ASSET_WRITERS", "SHARE_ASSET_ID"]].dropna().copy()
    compositions.loc[:, 'ASSET_WRITERS'] = compositions["ASSET_WRITERS"].str.replace("\n", " ", regex=False)
    compositions.loc[:, 'ASSET_WRITERS'] = compositions["ASSET_WRITERS"].str.replace(r'[\s]*,[\s]*\|', '|', regex=True)
    compositions.loc[:, 'ASSET_WRITERS'] = compositions["ASSET_WRITERS"].str.replace(r"','", '|', regex=True)
    compositions.loc[:, 'ASSET_WRITERS'] = compositions["ASSET_WRITERS"].str.replace(r",$", '', regex=True)
    comma_indices = compositions['ASSET_WRITERS'].str.contains(r",.*,") & \
        ~compositions['ASSET_WRITERS'].str.contains(r'[/\|;:]')
    compositions.loc[comma_indices, 'ASSET_WRITERS'] = compositions.loc[
        comma_indices, 'ASSET_WRITERS'].str.replace(",", "/")

    compositions.loc[:, 'ASSET_WRITERS'] = compositions['ASSET_WRITERS'].str.replace(r"[;:\|]", r" / ", regex=True)
    pat = re.compile(r"([\w\s().']+),[\s]*(.+?)(?=\s*(/|$))")

    for _ in range(5):
        compositions.loc[:, 'ASSET_WRITERS'] = compositions["ASSET_WRITERS"].str.replace(pat, r'\2 \1')
        if sum(compositions['ASSET_WRITERS'].str.contains(",", na=False)) == 0:
            break

    compositions.loc[:, 'ASSET_WRITERS'] = compositions["ASSET_WRITERS"].str.replace(r'\s[\s]+', ' ', regex=True)
    return compositions
