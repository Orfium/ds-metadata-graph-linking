from ds_metadata_graph_linking.trainer.callbacks.callback import CallbackHandler
from ds_metadata_graph_linking.trainer.callbacks.checkpoint import CheckpointCallback
from ds_metadata_graph_linking.trainer.callbacks.default_flow import DefaultFlowCallback
from ds_metadata_graph_linking.trainer.callbacks.early_stopping import EarlyStoppingCallback
from ds_metadata_graph_linking.trainer.callbacks.progress import ProgressCallback


def create_callback_handler(config, train_dataloader, validation_dataloader,
                            model_manager, report_manager, optimizer_manager, metrics_manager):
    default_callback = DefaultFlowCallback(model_manager, optimizer_manager, metrics_manager)
    earlystopping_callback = EarlyStoppingCallback(config.patience)
    checkpoint_callback = CheckpointCallback(config, model_manager, optimizer_manager)
    progress_callback = ProgressCallback(train_dataloader,
                                         validation_dataloader,
                                         report_manager,
                                         optimizer_manager,
                                         metrics_manager)

    callback_handler = CallbackHandler(callbacks=[default_callback,
                                                  earlystopping_callback,
                                                  checkpoint_callback,
                                                  progress_callback])

    return callback_handler
