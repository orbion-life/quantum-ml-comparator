"""
Built-in Datasets for QMC Benchmarks
=====================================
Provides standard datasets from sklearn for quantum vs classical ML
comparisons. No proprietary data -- all datasets are freely available.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DatasetMeta:
    """Metadata for a loaded dataset."""

    name: str
    n_features: int
    n_classes: int
    n_samples: int
    description: str


# Registry of available built-in datasets
_BUILTIN_DATASETS = {
    'iris': {
        'description': 'Iris flower classification (3 classes, 4 features)',
    },
    'breast_cancer': {
        'description': 'Breast cancer diagnosis (binary, 30 features)',
    },
    'wine': {
        'description': 'Wine cultivar classification (3 classes, 13 features)',
    },
    'digits': {
        'description': 'Handwritten digit classification (10 classes, 64 features)',
    },
    'moons': {
        'description': 'Two interleaving half-circles (binary, 2 features)',
    },
    'circles': {
        'description': 'Concentric circles (binary, 2 features)',
    },
    'blobs': {
        'description': 'Isotropic Gaussian blobs (3 classes, 2 features)',
    },
}


def list_datasets() -> List[str]:
    """Return list of available built-in dataset names."""
    return list(_BUILTIN_DATASETS.keys())


def load_dataset(name, test_size=0.3, random_state=42):
    """
    Load a built-in dataset, split into train/test.

    Parameters
    ----------
    name : str
        Dataset name. One of: "iris", "breast_cancer", "wine", "digits",
        "moons", "circles", "blobs".
    test_size : float
        Fraction of data for the test set (default: 0.3).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test, metadata)
        Where metadata is a DatasetMeta instance.

    Raises
    ------
    ValueError
        If dataset name is not recognized.
    """
    from sklearn.model_selection import train_test_split

    if name not in _BUILTIN_DATASETS:
        available = ', '.join(_BUILTIN_DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}"
        )

    X, y, desc = _load_raw(name, random_state)

    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    n_samples = X.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    meta = DatasetMeta(
        name=name,
        n_features=n_features,
        n_classes=n_classes,
        n_samples=n_samples,
        description=desc,
    )

    return X_train, X_test, y_train, y_test, meta


def load_from_csv(path, target_column, test_size=0.3, random_state=42):
    """
    Load a dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to CSV file.
    target_column : str
        Name of the target/label column.
    test_size : float
        Fraction of data for the test set.
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Columns: {list(df.columns)}"
        )

    y = df[target_column].values
    X = df.drop(columns=[target_column]).values.astype(np.float64)

    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    n_samples = X.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    meta = DatasetMeta(
        name=path,
        n_features=n_features,
        n_classes=n_classes,
        n_samples=n_samples,
        description=f"CSV dataset from {path}",
    )

    return X_train, X_test, y_train, y_test, meta


def load_from_arrays(X, y, test_size=0.3, random_state=42):
    """
    Split user-provided arrays into train/test.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Labels.
    test_size : float
        Fraction of data for the test set.
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    n_samples = X.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    meta = DatasetMeta(
        name="custom",
        n_features=n_features,
        n_classes=n_classes,
        n_samples=n_samples,
        description="User-provided arrays",
    )

    return X_train, X_test, y_train, y_test, meta


# ---------------------------------------------------------------------------
# Internal: load raw X, y for each built-in dataset
# ---------------------------------------------------------------------------

def _load_raw(name, random_state=42):
    """Load raw X, y arrays and description for a built-in dataset."""
    if name == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target, _BUILTIN_DATASETS[name]['description']

    elif name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        return data.data, data.target, _BUILTIN_DATASETS[name]['description']

    elif name == 'wine':
        from sklearn.datasets import load_wine
        data = load_wine()
        return data.data, data.target, _BUILTIN_DATASETS[name]['description']

    elif name == 'digits':
        from sklearn.datasets import load_digits
        data = load_digits()
        return data.data, data.target, _BUILTIN_DATASETS[name]['description']

    elif name == 'moons':
        from sklearn.datasets import make_moons
        X, y = make_moons(
            n_samples=1000, noise=0.2, random_state=random_state,
        )
        return X, y, _BUILTIN_DATASETS[name]['description']

    elif name == 'circles':
        from sklearn.datasets import make_circles
        X, y = make_circles(
            n_samples=1000, noise=0.1, factor=0.5,
            random_state=random_state,
        )
        return X, y, _BUILTIN_DATASETS[name]['description']

    elif name == 'blobs':
        from sklearn.datasets import make_blobs
        X, y = make_blobs(
            n_samples=1000, centers=3, n_features=2,
            random_state=random_state,
        )
        return X, y, _BUILTIN_DATASETS[name]['description']

    else:
        raise ValueError(f"No loader for dataset '{name}'")
