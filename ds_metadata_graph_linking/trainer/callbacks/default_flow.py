import torch

from ds_metadata_graph_linking.trainer.callbacks.callback import TrainerCallback


class DefaultFlowCallback(TrainerCallback):
    def __init__(self, model, optimizer, metrics):
        self.model = model
        self.metrics = metrics
        self.optimizer = optimizer

    def on_train_begin(self, config, state, control, **kwargs):
        control.reset()

    def on_train_step_end(self, config, state, control, **kwargs):
        control.should_log = state.current_step % config.log_every_n_steps == 0

        state.current_step += 1
        state.global_step += 1

    def on_validation_begin(self, config, state, control, **kwargs):
        self.model.eval()

    def on_validation_step_end(self, config, state, control, **kwargs):
        state.global_val_step += 1

    def on_epoch_begin(self, config, trainer_state, trainer_control, **kwargs):
        trainer_control.reset()
        self.model.train()
        self.metrics.reset()

    def on_epoch_end(self, config, state, control, **kwargs):
        torch.cuda.empty_cache()  # avoid cuda memory errors

        state.epochs_trained += 1
        state.current_step = 0  # resets every epoch but global does not

        validation_loss, _ = self.metrics.average_metrics('validation')

        state.current_validation_loss = validation_loss

        if state.epochs_trained >= config.epochs:
            control.should_training_stop = True

        if state.best_validation_loss > state.current_validation_loss:
            control.should_save = True
            state.best_validation_loss = state.current_validation_loss
