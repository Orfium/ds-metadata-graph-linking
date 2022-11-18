import torch
import os.path as osp

from torch_geometric.data import InMemoryDataset


class MetadataLinkingDataset(InMemoryDataset):
    def __init__(self, root, split, transform=None):
        super().__init__(root, transform, None)
        self.data = torch.load(osp.join(self.processed_dir, f'{split}_data.pt'))

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return ['node-feat', 'relations']

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please create {self.raw_file_names} and move it to '{self.raw_dir}'"
        )

    def process(self):
        raise RuntimeError(
            f"Processed dataset not found. "
            f"Please create {self.processed_file_names} and move it to '{self.processed_dir}'"
        )
