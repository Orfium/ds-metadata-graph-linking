import numpy as np
import pandas as pd
import os.path as osp
import fasttext.util
import fasttext as fastt

from tqdm import tqdm

from ds_metadata_graph_linking.featurizers.featurizer import Featurizer


class FastTextFeaturizer(Featurizer):
    def __init__(self):
        fasttext.util.download_model('en', if_exists='ignore')
        self.ft = fastt.load_model('cc.en.300.bin')

    def featurize(self, recordings, compositions, artists, isrcs, iswcs, clients, raw_graph_data):
        self.generate_features_recordings(recordings, raw_graph_data)
        self.generate_features_compositions(compositions, raw_graph_data)
        self.generate_features_artists(artists, raw_graph_data)
        self.generate_features_isrcs(isrcs, raw_graph_data)
        self.generate_features_iswcs(iswcs, raw_graph_data)
        self.generate_features_clients(clients, raw_graph_data)

    def generate_features_recordings(self, recordings: pd.DataFrame, raw_graph_data: str):
        recording_title_features = []

        for index, row in tqdm(recordings.iterrows(), position=0, leave=True):
            title = str(row['recording_title'])
            features = self.ft.get_sentence_vector(title)
            recording_title_features.append(features)

        recording_title_features_path = osp.join(raw_graph_data, 'recording_features.csv')
        recording_title_features_df = pd.DataFrame(np.array(recording_title_features))
        recording_title_features_df.to_csv(recording_title_features_path, index=False, header=None)

    def generate_features_compositions(self, compositions: pd.DataFrame, raw_graph_data: str):
        composition_title_features = []

        for index, row in tqdm(compositions.iterrows(), position=0, leave=True):
            title = str(row['composition_title'])
            features = self.ft.get_sentence_vector(title)
            composition_title_features.append(features)

        composition_title_features_path = osp.join(raw_graph_data, 'composition_features.csv')
        composition_title_features_df = pd.DataFrame(np.array(composition_title_features))
        composition_title_features_df.to_csv(composition_title_features_path, index=False, header=None)

    def generate_features_artists(self, artists: pd.DataFrame, raw_graph_data: str):
        artist_features = []

        for index, row in tqdm(artists.iterrows(), position=0, leave=True):
            title = str(row['name'])
            features = self.ft.get_sentence_vector(title)
            artist_features.append(features)

        artist_features_path = osp.join(raw_graph_data, 'artist_features.csv')
        artist_features_df = pd.DataFrame(np.array(artist_features))
        artist_features_df.to_csv(artist_features_path, index=False, header=None)

    def generate_features_isrcs(self, isrcs: pd.DataFrame, raw_graph_data: str):
        isrcs_features = []

        for index, row in tqdm(isrcs.iterrows(), position=0, leave=True):
            title = str(row['isrc'])
            features = self.ft.get_sentence_vector(title)
            isrcs_features.append(features)

        isrcs_features_path = osp.join(raw_graph_data, 'isrcs_features.csv')
        isrcs_features_df = pd.DataFrame(np.array(isrcs_features))
        isrcs_features_df.to_csv(isrcs_features_path, index=False, header=None)

    def generate_features_iswcs(self, iswcs: pd.DataFrame, raw_graph_data: str):
        iswcs_features = []

        for index, row in tqdm(iswcs.iterrows(), position=0, leave=True):
            title = str(row['iswc'])
            features = self.ft.get_sentence_vector(title)
            iswcs_features.append(features)

        iswcs_features_path = osp.join(raw_graph_data, 'iswcs_features.csv')
        iswcs_features_df = pd.DataFrame(np.array(iswcs_features))
        iswcs_features_df.to_csv(iswcs_features_path, index=False, header=None)

    def generate_features_clients(self, clients: pd.DataFrame, raw_graph_data: str):
        client_features = []

        for index, row in tqdm(clients.iterrows(), position=0, leave=True):
            title = str(row['client_name'])
            features = self.ft.get_sentence_vector(title)
            client_features.append(features)

        client_features_path = osp.join(raw_graph_data, 'clients_features.csv')
        client_features_df = pd.DataFrame(np.array(client_features))
        client_features_df.to_csv(client_features_path, index=False, header=None)
