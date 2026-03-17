import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


def get_probs(logits):
    preds = torch.argmax(logits, dim=1)
    probs = F.softmax(logits.float(), dim=1)
    return preds, probs


def get_metrics(preds, probs, targets):
    """Get classification metrics to log (acc, auroc)."""
    acc = balanced_accuracy_score(targets, preds)

    if probs.shape[1] == 2:
        auroc = roc_auc_score(targets, probs[:, 1])
    else:
        auroc = roc_auc_score(targets, probs, multi_class='ovr')

    return acc, auroc


def get_train_metrics(logits, targets):
    preds, probs = get_probs(logits)
    acc, auroc = get_metrics(preds, probs, targets)
    return acc, auroc


class EarlyStoppingLoss:
    """
    Args:
        patience (int): number of epochs to wait for improvement.
        min_delta (float): minimum change to qualify as an improvement.
        start_epoch (int): epoch to start the counter.
        model_file (str): filename to save the best model.
        verbose (bool): print a message when a checkpoint is saved and the training is stopped.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.1,
        start_epoch=0,
        model_file='model.pt',
        verbose=True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch
        self.model_file = model_file
        self.verbose = verbose

        self.best_metric = np.inf
        self.no_improvement_count = 0
        self.stop_training = False
        self.best_model = None

    def __call__(self, epoch, val_metric, checkpoint):
        if self.check_improvement(val_metric):
            # Modify best model and print stats
            self.best_model = checkpoint['state_dict']
            self.print_stats(val_metric)
            self.save_checkpoint(checkpoint)

            self.no_improvement_count = 0
            self.best_metric = val_metric
        else:
            # Best model is the one previously saved
            checkpoint['state_dict'] = self.best_model
            self.save_checkpoint(checkpoint)

            # Only start counting after start epoch
            if epoch >= self.start_epoch:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(
                        f'Stopping early after {self.patience} epochs with no improvement.'
                    )

    def check_improvement(self, val_metric):
        return val_metric < self.best_metric - self.min_delta

    def print_stats(self, val_metric):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.best_metric:.3f} → {val_metric:.3f}). Saving checkpoint...'
            )

    def save_checkpoint(self, checkpoint):
        """Save checkpoint when validation metric increases."""
        torch.save(checkpoint, self.model_file)
