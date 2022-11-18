import logging

import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Report:
    def __init__(self, metrics, progress_bar_writer=None):
        super(Report, self).__init__()
        self.metrics = metrics
        self.progress_bar_writer = progress_bar_writer

    @staticmethod
    def report_epoch(trainer_state, train_metrics, validation_metrics):
        wandb.log({
            'epoch': trainer_state.epochs_trained,
            f"train_loss": train_metrics[0],
            f"train_total_f1": train_metrics[1],
            f"val_loss": validation_metrics[0],
            f"val_total_f1": validation_metrics[1]
        })

        logger.info(f'\n'
                    f'Epoch {trainer_state.epochs_trained} finished. '
                    f'Early stopping counter: {trainer_state.early_stopping_counter} - '
                    f'train_loss: {train_metrics[0]} - '
                    f'train_total_f1: {train_metrics[1]} - '
                    f'val_loss: {validation_metrics[0]} - '
                    f'val_total_f1: {validation_metrics[1]}')

    def report_step(self, epoch, metrics, total_steps, mode='train'):
        wandb.log({f"{mode}_loss": metrics[0]}, step=total_steps)
        self.progress_bar_writer.set_description(f'Epoch {epoch} - '
                                                 f'{mode}_batch_loss:{metrics[0]} - '
                                                 f'{mode}_batch_f1:{metrics[1]}')

    def reset_progress_bar(self, dataloader):
        self.progress_bar_writer = tqdm(enumerate(dataloader), total=len(dataloader))
        return self.progress_bar_writer
