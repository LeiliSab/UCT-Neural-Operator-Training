# UFWI/io/load_mat.py

from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm


def load_mat(path):
    """
    Load variables from a MATLAB .mat file into a Python dict of NumPy arrays.

    Performs:
      - Transpose arrays with more than one dimension to match MATLAB ordering.
      - Convert arrays to Fortran (column-major) order for consistency.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .mat file.

    Returns
    -------
    dict
        Mapping from variable names (str) to NumPy arrays.
    """
    path = Path(path)
    data = {}
    print(f"Opening MAT file: {path}")
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        print(f"Found {len(keys)} variables: {keys}")
        for key in tqdm(keys, desc="Loading variables"):
            arr = np.array(f[key])
            if arr.ndim > 1:
                arr = arr.T
            arr = np.asfortranarray(arr)
            arr = arr.squeeze()
            data[key] = arr
            print(f"  • Loaded '{key}': shape {arr.shape}, dtype {arr.dtype}")
    print("Finished loading all variables.")
    return data
