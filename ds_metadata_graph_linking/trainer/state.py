import sys


class TrainerState:
    epochs_trained = 0
    current_step = 0
    global_step = 0
    global_val_step = 0
    num_train_epochs = 0
    early_stopping_counter = 0
    current_validation_loss = 0
    best_validation_loss = sys.float_info.max

    def load_from_checkpoint(self, **kwargs):
        self.epochs_trained = kwargs['epochs_trained']
        self.current_step = kwargs['current_step']
        self.global_step = kwargs['global_step']
        self.global_val_step = kwargs['global_val_step']
        self.early_stopping_counter = kwargs['early_stopping_counter']
        self.current_validation_loss = kwargs['current_validation_loss']
        self.best_validation_loss = kwargs['best_validation_loss']
