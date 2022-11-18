from sklearn.metrics import f1_score

from ds_metadata_graph_linking.utils.infer import infer_predictions_from_logits


class Metrics:
    def __init__(self, config):
        self.config = config
        self.steps = dict(train=0, validation=0)
        self.f1 = dict(train=0, validation=0)
        self.loss = dict(train=0, validation=0)
        self.labels = dict(train=[], validation=[])
        self.predictions = dict(train=[], validation=[])

    def update_metrics(self, logits, labels, loss=0, mode='train'):
        logits = logits.detach().cpu()  # to stop tracking gradients
        predictions = infer_predictions_from_logits(logits).numpy()
        labels = labels.detach().cpu().numpy()

        self.steps[mode] += 1
        self.f1[mode] += f1_score(labels, predictions)
        self.loss[mode] += loss
        self.labels[mode].extend(labels)
        self.predictions[mode].extend(predictions)

    def average_metrics(self, mode='train'):
        loss = self.loss[mode] / self.steps[mode]
        f1 = self.f1[mode] / self.steps[mode]

        return loss, f1

    def compute_metrics(self, mode='train'):
        loss = self.loss[mode] / self.steps[mode]
        return loss, f1_score(self.labels[mode], self.predictions[mode])

    def reset(self):
        self.steps = dict(train=0, validation=0)
        self.f1 = dict(train=0, validation=0)
        self.loss = dict(train=0, validation=0)
        self.labels = dict(train=[], validation=[])
        self.predictions = dict(train=[], validation=[])
