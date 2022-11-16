import torch


def infer_predictions_from_logits(logits, threshold=0.5):
    scores = infer_scores_from_logits(logits)
    return torch.where(scores > threshold, 1, 0)


def infer_scores_from_logits(logits):
    return torch.sigmoid(logits)
