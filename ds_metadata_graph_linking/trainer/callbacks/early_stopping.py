from ds_metadata_graph_linking.trainer.callbacks.callback import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=1):
        self.early_stopping_patience = early_stopping_patience

    def on_epoch_end(self, config, trainer_state, trainer_control, **kwargs):
        trainer_state.early_stopping_counter = trainer_state.early_stopping_counter + 1 \
            if trainer_state.current_validation_loss > trainer_state.best_validation_loss else 0

        if trainer_state.early_stopping_counter >= self.early_stopping_patience:
            trainer_control.should_training_stop = True
