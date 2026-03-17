import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import Subset
from torchvision import datasets, transforms

from wandb_example import utils

NORMALIZATION_MEAN = {'mnist': [0.1307]}
NORMALIZATION_SD = {'mnist': [0.3081]}


def get_augmentations(img_size, normalization=None):
    """Get augmentations for supervised training."""
    transform = {
        # Regular augmentations for imagenet
        'train': transforms.Compose(
            [
                transforms.RandomRotation(degrees=(-15, 15)),
                transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.1)),
                transforms.ToTensor(),
            ]
        ),
        'test': transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    }

    if normalization:
        normalize = transforms.Normalize(normalization['mean'], normalization['sd'])
        _ = [transform[k].transforms.append(normalize) for k in transform.keys()]

    return transform


def load_mnist(data_dir, transform, split='train', fold=0):
    if split != 'test':
        dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )

        # Load splits
        splits_file = Path(data_dir).joinpath('MNIST', 'splits.json')
        with open(splits_file, 'r') as f:
            idxs = json.load(f)[f'{fold}'][split]

        dataset = Subset(dataset, idxs)

    else:
        dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )

    return dataset


def generate_splits(
    data_dir,
    dataset,
    stratify=None,
    val_size=0.1,
    test_size=0.1,
    do_test_split=False,
    seed=42,
):
    """Generate splits for a dataset and save indices in a json file."""
    splits_file = Path(data_dir).joinpath('splits.json')
    idxs = np.arange(len(dataset))

    splits = {}
    # Generate test split
    idx_test = np.zeros(1)
    if do_test_split:
        val_size = val_size / (1 - test_size)
        idx_train, idx_test = train_test_split(
            idxs,
            test_size=test_size,
            shuffle=True,
            stratify=stratify,
            random_state=seed,
        )
        idxs = idx_train
        if stratify is not None:
            stratify = stratify[utils.get_matching_index(idxs, idx_train)]

    # Generate train and val splits
    idx_train, idx_val = train_test_split(
        idxs,
        test_size=val_size,
        shuffle=True,
        stratify=stratify,
        random_state=seed,
    )

    splits[0] = {
        'train': idx_train.tolist(),
        'val': idx_val.tolist(),
        'test': idx_test.tolist(),
    }

    print('train: ', idx_train.shape, 'val: ', idx_val.shape, 'test: ', idx_test.shape)

    # Generate 5 fold splits
    if stratify is None:
        kfold = KFold(5, shuffle=True, random_state=seed)
        kfold_splits = kfold.split(idxs)
    else:
        kfold = StratifiedKFold(5, shuffle=True, random_state=seed)
        kfold_splits = kfold.split(idxs, stratify)

    for i, (train_idx, val_idx) in enumerate(kfold_splits, start=1):
        splits[i] = {
            'train': idxs[train_idx].tolist(),
            'val': idxs[val_idx].tolist(),
        }

    print('train: ', train_idx.shape, 'val: ', val_idx.shape)

    with open(splits_file, 'w') as f:
        json.dump(splits, f)


if __name__ == '__main__':
    pass
