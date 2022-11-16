import torch.nn as nn


class CriterionManager:
    def __init__(self, config):
        self.loss = None
        self.config = config
        self.loss_function = nn.BCEWithLogitsLoss(reduction='mean')

    def update_loss(self, logits, labels):
        self.loss = self.loss_function(logits, labels)
        return self.loss.item()

    def update_gradients(self):
        self.loss.backward()
