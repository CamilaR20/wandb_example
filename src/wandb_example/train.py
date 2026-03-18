import os
import time
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from wandb_example import datasets, metrics, models, utils


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
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    return wandb_kwargs, checkpoint_file


def get_dataloaders(config):
    normalization = {
        'mean': datasets.NORMALIZATION_MEAN[config.dataset.name],
        'sd': datasets.NORMALIZATION_SD[config.dataset.name],
    }
    transform = datasets.get_augmentations(
        img_size=config.dataset.image_size, normalization=normalization
    )

    dataset_dir = hydra.utils.to_absolute_path(config.dataset_dir)
    dataset_train = datasets.load_mnist(
        dataset_dir, transform['train'], 'train', config.dataset.fold
    )
    dataset_val = datasets.load_mnist(
        dataset_dir, transform['test'], 'val', config.dataset.fold
    )
    n_classes = len(dataset_train.dataset.classes)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        drop_last=True,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        drop_last=True,
    )
    return dataloader_train, dataloader_val, n_classes


def get_optim(config, model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.wd,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=1)

    return optimizer, scheduler


def get_train_objs(config, checkpoint_file, n_classes, wandb_kwargs):
    # Load model, optimizer, scheduler and loss function
    start_epoch = 0

    if config.model.name == 'mlp':
        model = models.MLP(config.dataset.image_size, n_classes)
    else:
        model = models.CNN(config.model.dropout)

    optimizer, scheduler = get_optim(config, model)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
    stats = defaultdict(list)

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file, map_location=torch.device('cpu'), weights_only=False
        )

        start_epoch = checkpoint['epoch']
        stats = defaultdict(list, checkpoint['stats'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(config.device)

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        # Resume logging to the same id
        wandb_kwargs['id'] = checkpoint['wandb_id']
        wandb_kwargs['resume'] = 'must'

    return (model, optimizer, scheduler, loss_fn, start_epoch, stats, wandb_kwargs)


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.to(device)
    model.train()

    epoch_loss, epoch_acc, epoch_auroc = 0, 0, 0
    n_batches = len(dataloader)
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.type(torch.int64).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _, logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        acc, auroc = metrics.get_train_metrics(logits.detach().cpu(), y.detach().cpu())
        epoch_acc += acc
        epoch_auroc += auroc

    return epoch_loss / n_batches, epoch_acc / n_batches, epoch_auroc / n_batches


@torch.no_grad()
def validate_one_epoch(model, dataloader, loss_fn, device):
    model.to(device)
    model.eval()

    epoch_loss, epoch_acc, epoch_auroc = 0, 0, 0
    n_batches = len(dataloader)
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.type(torch.int64).to(device, non_blocking=True)

        _, logits = model(x)
        loss = loss_fn(logits, y)

        epoch_loss += loss.item()
        acc, auroc = metrics.get_train_metrics(logits.cpu(), y.cpu())
        epoch_acc += acc
        epoch_auroc += auroc

    return epoch_loss / n_batches, epoch_acc / n_batches, epoch_auroc / n_batches


@hydra.main(version_base=None, config_path='../../configs', config_name='default')
def main(config: DictConfig):
    utils.set_seed(config.seed)

    # Initialize checkpointing and wandb
    wandb_kwargs, checkpoint_file = init_experiment(config)

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader_train, dataloader_val, n_classes = get_dataloaders(config)
    model, optimizer, scheduler, loss_fn, start_epoch, stats, wandb_kwargs = (
        get_train_objs(config, checkpoint_file, n_classes, wandb_kwargs)
    )

    with wandb.init(**wandb_kwargs) as run:
        # Define metrics: x-axis, other metrics
        run.define_metric('epoch')
        run.define_metric('loss/*', step_metric='epoch')
        run.define_metric('acc/*', step_metric='epoch')
        run.define_metric('auroc/*', step_metric='epoch')

        # Watch weights and gradients
        run.watch(model, log=config.wandb.watch, log_freq=len(dataloader_train))

        # Early stopping saves the best model
        early_stopping = metrics.EarlyStoppingLoss(
            patience=config.stop.patience,
            min_delta=config.stop.min_delta,
            model_file=checkpoint_file,
            verbose=True,
        )

        for epoch in range(start_epoch, config.train.epochs):
            start_time = time.perf_counter()

            # Training
            train_loss, train_acc, train_auroc = train_one_epoch(
                model, dataloader_train, loss_fn, optimizer, config.device
            )
            stats['loss_train'].append(train_loss)
            stats['acc_train'].append(train_acc)
            stats['auroc_train'].append(train_auroc)

            # Validation
            val_loss, val_acc, val_auroc = validate_one_epoch(
                model, dataloader_val, loss_fn, config.device
            )
            stats['loss_val'].append(val_loss)
            stats['acc_val'].append(val_acc)
            stats['auroc_val'].append(val_auroc)
            scheduler.step()

            run.log(
                {
                    'epoch': epoch,
                    'loss/train': train_loss,
                    'acc/train': train_acc,
                    'auroc/train': train_auroc,
                    'loss/val': val_loss,
                    'acc/val': val_acc,
                    'auroc/val': val_auroc,
                }
            )

            # Early stopping
            checkpoint = {
                'state_dict': model.state_dict(),
                'backbone': config.model.name,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'stats': stats,
                'wandb_id': run.id,
            }
            early_stopping(epoch, val_loss, checkpoint)
            if early_stopping.stop_training:
                break

            end_time = time.perf_counter()
            print(
                f'Epoch {epoch + 1}, train loss {train_loss:.3f}, '
                f'val loss {val_loss:.3f}, val accuracy {val_acc:.3f}, {end_time - start_time:.1f} s'
            )


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print(f'Total time: {time.perf_counter() - start:.1f}s')
