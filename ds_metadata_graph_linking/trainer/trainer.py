import torch

from ds_metadata_graph_linking.model.model import Model
from ds_metadata_graph_linking.utils.edges import Edges
from ds_metadata_graph_linking.trainer.state import TrainerState
from ds_metadata_graph_linking.trainer.report import Report
from ds_metadata_graph_linking.trainer.control import TrainerControl
from ds_metadata_graph_linking.trainer.metrics import Metrics
from ds_metadata_graph_linking.trainer.criterion import Criterion
from ds_metadata_graph_linking.trainer.optimizer import Optimizer
from ds_metadata_graph_linking.trainer.callbacks.factory import create_callback_handler


class Trainer:

    def __init__(self, config, train_dataset, train_dataloader, validation_dataloader):
        self.config = config
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.criterion = Criterion(self.config)
        self.metrics = Metrics(self.config)
        self.report = Report(self.metrics)
        self.model = Model(self.config, self.train_dataset, self.train_dataloader)
        self.optimizer = Optimizer(config, train_dataloader, model_manager=self.model)

        self.state = TrainerState()
        self.control = TrainerControl()
        self.callback_handler = create_callback_handler(config, train_dataloader, validation_dataloader,
                                                        model_manager=self.model,
                                                        report_manager=self.report,
                                                        optimizer_manager=self.optimizer,
                                                        metrics_manager=self.metrics)

        self.state.num_train_epochs = self.config.epochs

    def train(self):
        self.callback_handler.on_train_begin(self.config, self.state, self.control)

        while self.control.continue_training():
            self.callback_handler.on_epoch_begin(self.config, self.state, self.control)
            self.train_epoch()
            self.validation_epoch()
            self.callback_handler.on_epoch_end(self.config, self.state, self.control)

    def train_epoch(self):
        train_progress_bar = self.report.reset_progress_bar(self.train_dataloader)
        for batch_step, batch in train_progress_bar:
            self.callback_handler.on_train_step_begin(self.config, self.state, self.control)
            self.train_step(batch)
            self.callback_handler.on_train_step_end(self.config, self.state, self.control)

    def train_step(self, batch):
        self.optimizer.reset()

        batch = batch.to(self.config.device)
        edge_to_predict_storage = batch[Edges.edge_to_predict]
        edge_label = edge_to_predict_storage.edge_label
        edge_label_index = edge_to_predict_storage.edge_label_index

        logits = self.model(x_dict=batch.x_dict,
                            edge_index_dict=batch.edge_index_dict,
                            edge_label_index=edge_label_index).view(-1)

        loss = self.criterion.update_loss(logits, edge_label)

        self.criterion.update_gradients()
        self.optimizer.step()

        self.criterion.empty_loss_cache()
        self.metrics.update_metrics(logits, edge_label, loss)

    def validation_epoch(self):
        self.callback_handler.on_validation_begin(self.config, self.state, self.control)

        validation_progress_bar = self.report.reset_progress_bar(self.validation_dataloader)
        for batch_step, batch in validation_progress_bar:
            self.validation_step(batch)
            self.callback_handler.on_validation_step_end(self.config, self.state, self.control)

    @torch.no_grad()
    def validation_step(self, batch):
        batch = batch.to(self.config.device)
        edge_to_predict_storage = batch[Edges.edge_to_predict]
        edge_label = edge_to_predict_storage.edge_label
        edge_label_index = edge_to_predict_storage.edge_label_index

        logits = self.model(x_dict=batch.x_dict,
                            edge_index_dict=batch.edge_index_dict,
                            edge_label_index=edge_label_index).view(-1)

        loss = self.criterion.update_loss(logits, edge_label)

        self.criterion.empty_loss_cache()
        self.metrics.update_metrics(logits, edge_label, loss, mode='validation')
