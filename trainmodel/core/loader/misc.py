import torch
import numpy as np
from importlib import metadata
import os
import json
from urllib.parse import unquote_plus, urlparse


def tensor_like(like, data):
    return torch.tensor(data, dtype=like.dtype, device=like.device)

def lrange(start, stop, step=1):
    return list(range(start, stop, step))

def resolve_data_path(data_path):
    if data_path != None: # Using custom data folder
        return data_path
    # Otherwise find the default folder

    # Fetch metadata saved by Pip
    url_file = metadata.distribution('noisebase').read_text('direct_url.json')

    if url_file == None: # Noisebase installed through PyPI
        editable = False
    else: # Noisebase installed from elsewhere (potentially editable)
        dist_info = json.loads(url_file)
        editable = dist_info.get('dir_info', {}).get('editable', False)

    if editable: # Default folder is cloned_repo/data
        nb_folder = unquote_plus(urlparse(dist_info['url']).path)
        return os.path.join(nb_folder, 'data')
    else: # Default folder is working_directory/data
        return os.path.join(os.getcwd(), 'data')

# Adapted from https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
def ACES(x, gamma_correct = True, gamma = 2.2):

    ACESInputMat = np.array([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ])

    ACESOutputMat = np.array([
        [1.60475, -0.53108, -0.07367],
        [-0.10208,  1.10813, -0.00605],
        [-0.00327, -0.07276,  1.07602]
    ])

    x = np.einsum('ji, hwi -> hwj', ACESInputMat, x)
    a = x * (x + 0.0245786) -  0.000090537
    b = x * (0.983729 * x + 0.4329510) + 0.238081
    x = a / b
    x = np.einsum('ji, hwi -> hwj', ACESOutputMat, x)

    if gamma_correct:
        return np.power(np.clip(x, 0.0, 1.0), 1.0/gamma)
    else:
        return x

class Shuffler():
    """Numpy rng convenience
    """
    def __init__(self, seed):
        rng = np.random.default_rng(seed = seed)
        self.seed = rng.integers(2**31)
    
    def shuffle(self, seed, sequence):
        rng = np.random.default_rng(seed = self.seed + seed)
        rng.shuffle(sequence)
    
    def split(self, seed, sequence, split, shuffled = True, smallest = 0):
        if shuffled:
            self.shuffle(seed, sequence)
        idx = max(round(len(sequence) * split), smallest)
        return sequence[:idx], sequence[idx:]
    
    def integers(self, seed, *args):
        rng = np.random.default_rng(seed = self.seed + seed)
        return rng.integers(*args)

    def derive(self, seed):
        return Shuffler(self.seed + seed)