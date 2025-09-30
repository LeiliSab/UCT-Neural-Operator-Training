# UFWI/io/save_results.py

from pathlib import Path
import scipy.io as sio


def save_results(path, results_dict):
    """
    Save a dictionary of arrays to a MATLAB .mat file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path for the .mat file.
    results_dict : dict
        Dictionary mapping variable names (str) to array-like objects.
    """
    path = Path(path)
    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use scipy.io.savemat with compression
    sio.savemat(str(path), results_dict, do_compression=True)
