from ds_metadata_graph_linking.trainer.callbacks.callback import TrainerCallback


class ProgressCallback(TrainerCallback):
    def __init__(self, train_dataloader, validation_dataloader, report_manager, optimizer_manager, metrics_manager):
        self.report_manager = report_manager
        self.metrics_manager = metrics_manager
        self.optimizer_manager = optimizer_manager
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def on_train_step_end(self, config, state, control, **kwargs):
        if control.should_log:
            train_metrics = self.metrics_manager.average_metrics(mode='train')
            self.report_manager.report_step(state.epochs_trained, train_metrics, state.global_step, mode='train')

    def on_validation_step_end(self, config, state, control, **kwargs):
        validation_metrics = self.metrics_manager.average_metrics(mode='validation')
        self.report_manager.report_step(state.epochs_trained, validation_metrics,
                                        state.global_val_step, mode='validation')

    def on_epoch_end(self, config, trainer_state, trainer_control, **kwargs):
        train_metrics = self.metrics_manager.compute_metrics(mode='train')
        validation_metrics = self.metrics_manager.compute_metrics(mode='validation')

        self.report_manager.report_epoch(trainer_state, train_metrics, validation_metrics)
