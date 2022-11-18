from enum import Enum


class Files(Enum):
    config = 'config.yml'
    model = 'model.bin'
    optimizer = 'optimizer.bin'
    metadata = 'model_metadata.json'
