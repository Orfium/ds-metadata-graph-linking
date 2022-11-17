import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from ds_metadata_graph_linking.utils.edges import Edges
from ds_metadata_graph_linking.utils.infer import infer_scores_from_logits, infer_predictions_from_logits


def train(config, train_dataloader, val_dataloader,
          val_data, test_data, model, optimizer, criterion, checkpoints_path):
    train_total_steps = 0
    val_total_steps = 0
    for epoch in range(1, config.epochs):
        train_total_steps, train_loss, train_total_f1 = train_epoch(train_total_steps, epoch, config,
                                                                    model, optimizer, criterion,
                                                                    train_dataloader)
        val_total_steps, val_loader_loss, val_loader_f1 = val_epoch(val_total_steps, epoch, config,
                                                                    model, criterion,
                                                                    val_dataloader)

        test_labels, test_logits, test_loss = test(config, test_data, model, criterion)
        val_labels, val_logits, val_loss = test(config, val_data, model, criterion)

        report_epoch(epoch, train_loss, train_total_f1, val_loader_loss, val_loader_f1,
                     val_labels, val_logits, val_loss,
                     test_labels, test_logits, test_loss)

        save(checkpoints_path, model, optimizer)


def train_epoch(total_steps, epoch, config, model, optimizer, criterion, train_dataloader):
    model.train()
    total_examples = total_loss = total_f1_score = 0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for index, batch in progress_bar:
        batch = batch.to(config.device)
        edge_to_predict_storage = batch[Edges.edge_to_predict]
        edge_label = edge_to_predict_storage.edge_label
        edge_label_index = edge_to_predict_storage.edge_label_index

        optimizer.reset()

        logits = model(x_dict=batch.x_dict,
                       edge_index_dict=batch.edge_index_dict,
                       edge_label_index=edge_label_index).view(-1)

        loss = criterion.update_loss(logits, edge_label)
        criterion.update_gradients()
        optimizer.step()

        logits = logits.detach().cpu()  # to stop tracking gradients
        predictions = infer_predictions_from_logits(logits).numpy()
        labels = edge_label.detach().cpu().numpy()
        total_f1_score += f1_score(labels, predictions)

        total_steps += 1
        total_examples += 1
        total_loss += float(loss)

        report_step(config, epoch, index, total_loss, total_f1_score,
                    total_examples, total_steps, progress_bar)

    criterion.empty_loss_cache()
    torch.cuda.empty_cache()  # avoid cuda memory errors

    return total_steps, total_loss / total_examples, total_f1_score / total_examples


@torch.no_grad()
def val_epoch(total_steps, epoch, config, model, criterion, val_dataloader):
    model.eval()
    total_examples = total_loss = total_f1_score = 0
    progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    for index, batch in progress_bar:
        batch = batch.to(config.device)
        edge_to_predict_storage = batch[Edges.edge_to_predict]
        edge_label = edge_to_predict_storage.edge_label
        edge_label_index = edge_to_predict_storage.edge_label_index

        logits = model(x_dict=batch.x_dict,
                       edge_index_dict=batch.edge_index_dict,
                       edge_label_index=edge_label_index).view(-1)

        loss = criterion.update_loss(logits, edge_label)

        labels = edge_label.detach().cpu().numpy()
        predictions = infer_predictions_from_logits(logits).detach().cpu().numpy()
        total_f1_score += f1_score(labels, predictions)

        total_examples += 1
        total_loss += float(loss)

        if index % config.log_every == 0:
            wandb.log({"val_loss": total_loss / total_examples}, step=total_steps)
            progress_bar.set_description(f'Epoch {epoch} - '
                                         f'val_batch_loss:{total_loss / total_examples} - '
                                         f'val_batch_f1:{total_f1_score / total_examples}')

        total_steps += 1

    criterion.empty_loss_cache()
    torch.cuda.empty_cache()  # avoid cuda memory errors

    return total_steps, total_loss / total_examples, total_f1_score / total_examples


@torch.no_grad()
def test(config, data, model, criterion):
    model.eval()

    data.to(config.device)
    edge_to_predict_storage = data[Edges.edge_to_predict]
    edge_label = edge_to_predict_storage.edge_label
    edge_label_index = edge_to_predict_storage.edge_label_index

    logits = model(x_dict=data.x_dict,
                   edge_index_dict=data.edge_index_dict,
                   edge_label_index=edge_label_index).view(-1)

    loss = criterion.update_loss(logits, edge_label)

    labels = edge_label.cpu().numpy()
    scores = infer_scores_from_logits(logits).cpu().numpy()

    criterion.empty_loss_cache()
    torch.cuda.empty_cache()  # avoid cuda memory errors

    return labels, scores, loss


def save(checkpoints_path, model, optimizer):
    model_state_dict = model.state()
    model_state_dict_path = os.path.join(checkpoints_path, 'model.pt')
    torch.save(model_state_dict, model_state_dict_path)
    print('Model saved to ', model_state_dict_path)

    optimizer_state_dict = optimizer.state()
    optimizer_state_dict_path = os.path.join(checkpoints_path, 'optimizer.pt')
    torch.save(optimizer_state_dict, optimizer_state_dict_path)
    print('Optimizer saved to ', optimizer_state_dict_path)


def report_step(config, epoch, index, total_loss, total_f1_score, total_examples, total_steps, progress_bar):
    if index % config.log_every == 0:
        wandb.log({"train_loss": total_loss / total_examples}, step=total_steps)
        progress_bar.set_description(f'Epoch {epoch} - '
                                     f'train_batch_loss:{total_loss / total_examples} - '
                                     f'train_batch_f1:{total_f1_score / total_examples}')


def report_epoch(epoch, train_loss,
                 train_total_f1, val_loader_loss, val_loader_f1,
                 val_labels, val_logits, val_loss,
                 test_labels, test_logits, test_loss):
    test_preds = np.where(test_logits > 0.5, 1, 0)
    val_preds = np.where(val_logits > 0.5, 1, 0)

    wandb.sklearn.plot_confusion_matrix(test_labels, test_preds, [0, 1])

    wandb.log({
        'epoch': epoch,
        f"train_loss": float(train_loss),
        f"train_total_f1": train_total_f1,
        f"val_loader_f1": val_loader_f1,
        f"val_loader_loss": float(val_loader_loss),
        f"val_loss": float(val_loss),
        f"test_loss": float(test_loss),
        f"val_f1_score": float(f1_score(val_labels, val_preds)),
        f"test_f1_score": float(f1_score(test_labels, test_preds)),
        f'val_recall_score': float(recall_score(val_labels, val_preds)),
        f'val_precision_score': float(precision_score(val_labels, val_preds)),
        f'test_recall_score': float(recall_score(test_labels, test_preds)),
        f'test_precision_score': float(precision_score(test_labels, test_preds)),
    })
