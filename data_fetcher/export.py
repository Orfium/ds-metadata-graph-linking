import os.path
from enum import Enum
from typing import Union

import pandas as pd


class NodePath(Enum):
    Recording = "recording.csv"
    ISRC = "isrc.csv"
    Composition = "composition.csv"
    ISWC = "iswc.csv"
    HFA_CODE = "hfa_code.csv"
    Artist = "artist.csv"
    Client = "client.csv"


class RelPath(Enum):
    HAS_ISRC = "has_isrc.csv"
    PERFORMED = "performed.csv"
    EMBEDDED = "embedded.csv"
    HAS_ISWC = "has_iswc.csv"
    HAS_HFA_CODE = "has_hfa_code.csv"
    WROTE = 'wrote.csv'
    OWNS = 'owns.csv'


def export_csv(df: pd.DataFrame, path: Union[NodePath, RelPath], raw_data_path: str):
    df = df.copy()
    path = os.path.join(raw_data_path, path.value)
    df.to_csv(path, index=False, header=False, mode='a')
