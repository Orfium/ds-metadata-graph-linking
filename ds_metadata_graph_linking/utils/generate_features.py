import fasttext as fastt
import fasttext.util
import pandas as pd
from tqdm import tqdm
import numpy as np

fasttext.util.download_model('en', if_exists='ignore')
ft = fastt.load_model('cc.en.300.bin')


def generate_features_recordings(recordings: pd.DataFrame) -> pd.DataFrame:
    recording_title_features = []

    for index, row in tqdm(recordings.iterrows(), position=0, leave=True):
        title = str(row['recording_title'])
        features = ft.get_sentence_vector(title)
        recording_title_features.append(features)

    pd.DataFrame(np.array(recording_title_features)).to_csv(
        'data/base_dataset/processed_dataset/recording_features.csv', index=False, header=None)
    features = pd.read_csv('data/base_dataset/processed_dataset/recording_features.csv', header=None)
    return features


def generate_features_compositions(compositions: pd.DataFrame) -> pd.DataFrame:
    composition_title_features = []

    for index, row in tqdm(compositions.iterrows(), position=0, leave=True):
        title = str(row['composition_title'])
        features = ft.get_sentence_vector(title)
        composition_title_features.append(features)

    pd.DataFrame(np.array(composition_title_features)).to_csv(
        'data/base_dataset/processed_dataset/composition_features.csv', index=False, header=None)
    features = pd.read_csv('data/base_dataset/processed_dataset/composition_features.csv', header=None)
    return features


def generate_features_artists(artists: pd.DataFrame) -> pd.DataFrame:
    artist_features = []

    for index, row in tqdm(artists.iterrows(), position=0, leave=True):
        title = str(row['name'])
        features = ft.get_sentence_vector(title)
        artist_features.append(features)

    pd.DataFrame(np.array(artist_features)).to_csv('data/base_dataset/processed_dataset/artist_features.csv',
                                                   index=False, header=None)
    features = pd.read_csv('data/base_dataset/processed_dataset/artist_features.csv', header=None)
    return features


def generate_features_isrcs(isrcs: pd.DataFrame) -> pd.DataFrame:
    isrcs_features = []

    for index, row in tqdm(isrcs.iterrows(), position=0, leave=True):
        title = str(row['isrc'])
        features = ft.get_sentence_vector(title)
        isrcs_features.append(features)

    pd.DataFrame(np.array(isrcs_features)).to_csv('data/base_dataset/processed_dataset/isrcs_features.csv', index=False,
                                                  header=None)
    features = pd.read_csv('data/base_dataset/processed_dataset/isrcs_features.csv', header=None)
    return features


def generate_features_iswcs(iswcs: pd.DataFrame) -> pd.DataFrame:
    iswcs_features = []

    for index, row in tqdm(iswcs.iterrows(), position=0, leave=True):
        title = str(row['iswc'])
        features = ft.get_sentence_vector(title)
        iswcs_features.append(features)

    pd.DataFrame(np.array(iswcs_features)).to_csv('data/base_dataset/processed_dataset/iswcs_features.csv', index=False,
                                                  header=None)
    features = pd.read_csv('data/base_dataset/processed_dataset/iswcs_features.csv', header=None)
    return features


def generate_features_clients(clients: pd.DataFrame) -> pd.DataFrame:
    client_features = []

    for index, row in tqdm(clients.iterrows(), position=0, leave=True):
        title = str(row['client_name'])
        features = ft.get_sentence_vector(title)
        client_features.append(features)

    pd.DataFrame(np.array(client_features)).to_csv('data/base_dataset/processed_dataset/clients_features.csv',
                                                   index=False, header=None)
    features = pd.read_csv('data/base_dataset/processed_dataset/clients_features.csv', header=None)
    return features
