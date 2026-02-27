from __future__ import annotations

import numpy as np
from pathlib import Path

def load_mnist(data_dir: str = './data/mnist') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset from local CSVs or fallback to scikit-learn fetchers.
    
    Returns:
        x_train, y_train, x_test, y_test
    """
    local_train = Path(data_dir) / 'mnist_train.csv'
    local_test = Path(data_dir) / 'mnist_test.csv'
    
    if local_train.exists() and local_test.exists():
        train_data = np.loadtxt(local_train, delimiter=',')
        test_data = np.loadtxt(local_test, delimiter=',')
        
        y_train = train_data[:, 0].astype(np.int64)
        x_train = train_data[:, 1:]
        y_test = test_data[:, 0].astype(np.int64)
        x_test = test_data[:, 1:]
    else:
        from sklearn.datasets import fetch_openml, load_digits
        try:
            x, y = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False, return_X_y=True)
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.int64)
            x_train = x[:60000]
            y_train = y[:60000]
            x_test = x[60000:]
            y_test = y[60000:]
        except Exception:
            x, y = load_digits(return_X_y=True)
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.int64)
            x = x.reshape(-1, 8, 8)
            x = np.pad(x, ((0, 0), (10, 10), (10, 10)), mode='constant').reshape(-1, 784)
            split = int(0.8 * x.shape[0])
            x_train = x[:split]
            y_train = y[:split]
            x_test = x[split:]
            y_test = y[split:]
            
    return x_train, y_train, x_test, y_test
