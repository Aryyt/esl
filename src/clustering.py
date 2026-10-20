from .model import Model

class KNeighborsRegressor(Model):
    
    def __init__(self, dist_func=None):
        super().__init__()
        self.distance_mat = None
        self.dist_func = dist_func