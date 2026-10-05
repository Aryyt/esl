import numpy as np
from numpy.typing import ArrayLike

from .model import Model

class LinearRegression(Model):
    def __init__(self):
        super().__init__()
    
    def fit(x: ArrayLike, y: ArrayLike):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        
        pass
        