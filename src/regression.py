import numpy as np
from numpy.typing import ArrayLike, NDArray
from .model import Model

class LinearRegression(Model):
    def __init__(self):
        super().__init__()
        
        self.coefs_ = None
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        X: NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        y: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        
        X = np.column_stack((np.ones(X.shape[0], dtype=np.float64), X))
        
        assert X.ndim == 2, "X must have shape (n_samples, n_features)"
        assert y.ndim == 1, "y must have shape (n_samples,)"
        assert X.shape[0] == y.shape[0], "X and y must contain the same number of samples"
        
        print(X)
        
        self.coefs_ = np.linalg.solve((X.T @ X), X.T @ y)
        
    def predict(self, X: ArrayLike):
        X: NDArray[np.float64] = np.asarray(X, dtype=np.float64)
        X = np.column_stack((np.ones(X.shape[0], dtype=np.float64), X))
        assert X.shape[1] == self.coefs_.shape[0], "Shape missmatch"        
        
        return X @ self.coefs_
        