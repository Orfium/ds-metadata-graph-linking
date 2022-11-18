import os
import json
import torch

from ds_metadata_graph_linking.trainer.callbacks.callback import TrainerCallback
from ds_metadata_graph_linking.utils.files import Files


class CheckpointCallback(TrainerCallback):
    def __init__(self, config, model_manager, optimizer_manager):
        self.model_manager = model_manager
        self.optimizer_manager = optimizer_manager
        self.checkpoint_dir = config.checkpoints_path

    def _save_optimizer(self):
        optimizer_state = self.optimizer_manager.state()
        optimizer_to_save_path = os.path.join(self.checkpoint_dir, Files.optimizer.value)

        torch.save(optimizer_state, optimizer_to_save_path)

    def _save_metadata(self, config, state):
        metadata_to_save = {
            'train_state': vars(state),
            'train_config': vars(config)
        }

        metadata_to_save_path = os.path.join(self.checkpoint_dir, Files.metadata.value)

        with open(metadata_to_save_path, "w") as metadata_file:
            json.dump(metadata_to_save, metadata_file, indent=4, sort_keys=False)

    def _save_checkpoint(self):
        model_state = self.model_manager.state()
        checkpoint_to_save_path = os.path.join(self.checkpoint_dir, Files.model.value)

        torch.save(model_state, checkpoint_to_save_path)

    def _load_optimizer(self, config):
        optimizer_to_load_path = os.path.join(self.checkpoint_dir, Files.optimizer.value)
        optimizer_state = torch.load(optimizer_to_load_path, map_location=config.device)

        self.optimizer_manager.load_from_checkpoint(optimizer_state)

    def _load_metadata(self, state):
        metadata_to_load_path = os.path.join(self.checkpoint_dir, Files.metadata.value)

        with open(metadata_to_load_path, "rb") as metadata_file:
            metadata = json.load(metadata_file)

        state.load_from_checkpoint(**metadata['train_state'])

    def _load_checkpoint(self, config):
        checkpoint_file = os.path.join(os.path.abspath(self.checkpoint_dir), Files.model.value)
        model_state = torch.load(checkpoint_file, map_location=config.device)

        self.model_manager.load_from_checkpoint(model_state)

    def on_train_begin(self, config, state, control, **kwargs):
        if config.resume:
            self._load_checkpoint(config)
            self._load_optimizer(config)
            self._load_metadata(state)

    def on_epoch_end(self, config, state, control, **kwargs):
        if control.should_save:
            self._save_optimizer()
            self._save_checkpoint()
            self._save_metadata(config, state)
