import json
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from wandb_example import datasets, models, utils


def init_experiment(config):
    hydra_cfg = HydraConfig.get()
    experiment_dir = Path(hydra_cfg.runtime.output_dir)

    # Check if it is a sweep
    is_sweep = hydra_cfg.mode.name == 'MULTIRUN'

    # Args to initialize wandb
    if is_sweep:
        run_name = hydra_cfg.job.override_dirname
        group = config.sweep_name
    else:
        run_name = config.experiment_name
        group = config.experiment_name

    wandb_kwargs = dict(
        project='wandb_example',
        dir=experiment_dir,  # hydra run directory
        name=run_name,
        group=group,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.wandb.mode,
        settings=wandb.Settings(
            console='off',
            save_code=False,  # no code
            disable_git=True,  # no git diffs
            _disable_meta=True,  # no metadata
        ),
    )

    # Initialize checkpoint file separate from logs
    project_dir = Path.cwd()
    checkpoint_file = project_dir.joinpath('checkpoints', group, f'{run_name}.pt')
    results_file = project_dir.joinpath('checkpoints', group, f'{run_name}.json')
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    return wandb_kwargs, checkpoint_file, results_file


def get_dataloaders(config):
    normalization = {
        'mean': datasets.NORMALIZATION_MEAN[config.dataset.name],
        'sd': datasets.NORMALIZATION_SD[config.dataset.name],
    }
    transform = datasets.get_augmentations(
        img_size=config.dataset.image_size, normalization=normalization
    )

    dataset_dir = hydra.utils.to_absolute_path(config.dataset_dir)
    dataset_test = datasets.load_mnist(dataset_dir, transform['test'], 'test')
    n_classes = len(dataset_test.classes)

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        drop_last=False,
    )
    return dataloader_test, n_classes


def get_model(config, checkpoint_file, n_classes, wandb_kwargs):
    if config.model.name == 'mlp':
        model = models.MLP(config.dataset.image_size, n_classes)
    else:
        model = models.CNN(config.model.dropout)

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file, map_location=torch.device('cpu'), weights_only=False
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(config.device)

        # Resume logging to the same id
        wandb_kwargs['id'] = checkpoint['wandb_id']
        wandb_kwargs['resume'] = 'must'
    else:
        raise Exception(f'Checkpoint file does not exist: {checkpoint_file}')

    return model, wandb_kwargs


@torch.no_grad()
def predict(model, dataloader, device):
    model.to(device)
    model.eval()

    preds, probs, targets, embeddings = [], [], [], []
    for b, (img, labels) in enumerate(tqdm(dataloader)):
        img, labels = img.to(device), labels.to(device)

        h, y = model(img)
        pred = torch.argmax(y, dim=1)
        p = F.softmax(y, dim=1)

        preds.extend(pred.tolist())
        probs.extend(p.tolist())
        targets.extend(labels.tolist())
        embeddings.extend(h.tolist())

    preds = np.array(preds)
    probs = np.array(probs)
    targets = np.array(targets)
    embeddings = np.array(embeddings)

    return preds, probs, targets, embeddings


def get_all_metrics(targets, preds, probs):
    acc = balanced_accuracy_score(targets, preds)
    kappa = cohen_kappa_score(targets, preds, weights='quadratic')

    if probs.ndim == 1:
        auroc = roc_auc_score(targets, probs)
        auprc = average_precision_score(targets, probs)
    elif probs.shape[1] == 2:
        auroc = roc_auc_score(targets, probs[:, 1])
        auprc = average_precision_score(targets, probs[:, 1])
    else:
        auroc = roc_auc_score(targets, probs, multi_class='ovr')
        auprc = average_precision_score(targets, probs, average='macro')

    return auroc, auprc, acc, kappa


def load_eval(checkpoints_dir, experiment_name):
    # Load results
    results_file = checkpoints_dir.joinpath(f'{experiment_name}.json')
    with open(results_file, 'r') as f:
        results = json.load(f)

    targets, preds, probs = [
        np.array(results[k]) for k in ['targets', 'preds', 'probs']
    ]
    return targets, preds, probs


@hydra.main(version_base=None, config_path='../../configs', config_name='default')
def main(config: DictConfig):
    utils.set_seed(config.seed)

    # Initialize checkpointing and wandb
    wandb_kwargs, checkpoint_file, results_file = init_experiment(config)

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader_test, n_classes = get_dataloaders(config)
    model, wandb_kwargs = get_model(config, checkpoint_file, n_classes, wandb_kwargs)

    with wandb.init(**wandb_kwargs) as run:
        # Check for dead convolutional layers
        dead_layer_count = 0
        for name, parameters in model.named_parameters():
            if 'conv' in name:
                max_weight = parameters.flatten().abs().max()

                if max_weight <= 1e-4:
                    dead_layer_count += 1

        print(f'Dead layer count (max(abs(parameters) <= 1e-4 ) = {dead_layer_count}')

        # Get predictions (embeddings from the projector)
        preds_test, probs_test, targets_test, embeddings_test = predict(
            model, dataloader_test, config.device
        )

        # Metrics
        auroc, auprc, acc, kappa = get_all_metrics(targets_test, preds_test, probs_test)
        run.summary['auroc/eval'] = auroc
        run.summary['auprc/eval'] = auprc
        run.summary['acc/eval'] = acc
        run.summary['kappa/eval'] = kappa

        # Save results
        results = {
            'embeddings': embeddings_test.tolist(),
            'targets': targets_test.tolist(),
            'preds': preds_test.tolist(),
            'probs': probs_test.tolist(),
        }

        with open(results_file, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print(f'Total time: {time.perf_counter() - start:.1f}s')
