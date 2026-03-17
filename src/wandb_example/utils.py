import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_image(filename, return_np=True):
    """Load image file as a PIL image or a numpy array."""
    image = Image.open(filename)
    image.load()
    data = np.asarray(image, dtype='int32')
    return data if return_np else image


def get_matching_index(arr_ref, arr_match):
    """Get indices for elements in arr_ref that match elements in arr_match."""
    df = pd.DataFrame({'Value': np.arange(arr_ref.size)}, index=arr_ref)
    arr_match = arr_match[np.isin(arr_match, arr_ref)]
    df = df.loc[arr_match]
    return df['Value'].to_numpy()
