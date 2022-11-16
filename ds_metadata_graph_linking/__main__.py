import os
import click
import torch
import wandb
import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from ds_metadata_graph_linking.dataset.factory import create_dataloader, create_dataset, split_data
from ds_metadata_graph_linking.model.manager import ModelManager
from ds_metadata_graph_linking.trainer.config import load_config
from ds_metadata_graph_linking.trainer.criterion import CriterionManager
from ds_metadata_graph_linking.trainer.optimizer import OptimizerManager
from ds_metadata_graph_linking.utils.infer import infer_scores_from_logits, infer_predictions_from_logits
from ds_metadata_graph_linking.utils.train import set_seed

wandb.init(project="link_prediction", entity="stavros-giorgis", name="r2c_test")


@click.group()
def cli():
    pass


def train(total_steps, epoch, config, model, optimizer, criterion, train_dataloader, e_to_predict):
    model.train()
    total_examples = total_loss = total_f1_score = 0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for index, batch in progress_bar:
        batch = batch.to(config.device)
        optimizer.reset()
        paper_cites_paper_storage = batch[e_to_predict]

        logits = model(x_dict=batch.x_dict,
                       edge_index_dict=batch.edge_index_dict,
                       edge_label_index=paper_cites_paper_storage.edge_label_index).view(-1)

        loss = criterion.update_loss(logits, paper_cites_paper_storage.edge_label)
        criterion.update_gradients()

        optimizer.step()

        total_examples += 1
        total_loss += float(loss)

        logits = logits.detach().cpu()  # to stop tracking gradients
        predictions = infer_predictions_from_logits(logits).numpy()
        labels = paper_cites_paper_storage.edge_label.detach().cpu().numpy()
        total_f1_score += f1_score(labels, predictions)

        if index % config.log_every == 0:
            wandb.log({"train_loss": total_loss / total_examples}, step=total_steps)
            progress_bar.set_description(f'Epoch {epoch} - '
                                         f'train_batch_loss:{total_loss / total_examples} - '
                                         f'train_batch_f1:{total_f1_score / total_examples}')

        total_steps += 1

    torch.cuda.empty_cache()

    return total_steps, total_loss / total_examples, total_f1_score / total_examples


@torch.no_grad()
def val(total_steps, epoch, config, model, criterion, val_dataloader, e_to_predict):
    model.eval()
    total_examples = total_loss = total_f1_score = 0
    progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    for index, batch in progress_bar:
        batch = batch.to(config.device)
        paper_cites_paper_storage = batch[e_to_predict]

        logits = model(x_dict=batch.x_dict,
                       edge_index_dict=batch.edge_index_dict,
                       edge_label_index=paper_cites_paper_storage.edge_label_index).view(-1)

        loss = criterion.update_loss(logits, paper_cites_paper_storage.edge_label)

        total_examples += 1
        total_loss += float(loss)

        predictions = infer_predictions_from_logits(logits).detach().cpu().numpy()
        labels = paper_cites_paper_storage.edge_label.detach().cpu().numpy()
        total_f1_score += f1_score(labels, predictions)

        if index % config.log_every == 0:
            wandb.log({"val_loss": total_loss / total_examples}, step=total_steps)
            progress_bar.set_description(f'Epoch {epoch} - '
                                         f'val_batch_loss:{total_loss / total_examples} - '
                                         f'val_batch_f1:{total_f1_score / total_examples}')

        total_steps += 1

    torch.cuda.empty_cache()

    return total_steps, total_loss / total_examples, total_f1_score / total_examples


@torch.no_grad()
def test(config, data, model, criterion, e_to_predict):
    model.eval()
    data.to(config.device)
    paper_cites_paper_storage = data[e_to_predict]

    logits = model(x_dict=data.x_dict,
                   edge_index_dict=data.edge_index_dict,
                   edge_label_index=paper_cites_paper_storage.edge_label_index).view(-1)

    loss = criterion.update_loss(logits, paper_cites_paper_storage.edge_label)

    scores = infer_scores_from_logits(logits).cpu().numpy()
    labels = paper_cites_paper_storage.edge_label.cpu().numpy()

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


def report_results(epoch, train_loss,
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


@cli.command(name='train')
@click.option('--config', type=click.STRING, required=True)
@click.option('--dataset_path', type=click.STRING, required=True)
@click.option('--checkpoints_path', type=click.STRING, required=True)
@click.option('--debug/--no-debug', type=click.BOOL, default=False)
def train_entrypoint(config, dataset_path, checkpoints_path, debug):
    config = load_config(config, dataset_path, checkpoints_path)
    set_seed(config.seed)

    dataset = create_dataset(dataset_path, debug=debug)

    e_to_predict = ('composition', 'embedded', 'recording')
    re_to_predict = ('recording', 'rev_embedded', 'composition')

    train_data, val_data, test_data = split_data(config, dataset, e_to_predict, re_to_predict)

    train_dataloader = create_dataloader(config=config,
                                         dataset=train_data,
                                         edge_label=None,
                                         neg_sampling_ratio=config.neighbor_loader_neg_sampling_ratio,
                                         edge_label_index=(e_to_predict, train_data[e_to_predict].edge_label_index))

    val_dataloader = create_dataloader(config=config,
                                       dataset=val_data,
                                       neg_sampling_ratio=0,  # no need to sample extra negative edges
                                       edge_label=val_data[e_to_predict].edge_label,
                                       edge_label_index=(e_to_predict, val_data[e_to_predict].edge_label_index))

    model = ModelManager(config, train_data, train_dataloader, e_to_predict)
    optimizer = OptimizerManager(config, train_dataloader, model_manager=model)
    criterion = CriterionManager(config)

    train_total_steps = 0
    val_total_steps = 0
    for epoch in range(1, config.epochs):
        train_total_steps, train_loss, train_total_f1 = train(train_total_steps, epoch, config,
                                                              model, optimizer, criterion,
                                                              train_dataloader, e_to_predict)
        val_total_steps, val_loader_loss, val_loader_f1 = val(val_total_steps, epoch, config,
                                                              model, criterion,
                                                              val_dataloader, e_to_predict)

        test_labels, test_logits, test_loss = test(config, test_data, model, criterion, e_to_predict)
        val_labels, val_logits, val_loss = test(config, val_data, model, criterion, e_to_predict)

        report_results(epoch, train_loss, train_total_f1, val_loader_loss, val_loader_f1,
                       val_labels, val_logits, val_loss,
                       test_labels, test_logits, test_loss)

        save(checkpoints_path, model, optimizer)


if __name__ == '__main__':
    cli()
