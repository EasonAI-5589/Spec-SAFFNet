import numpy as np
from sklearn.metrics import mean_squared_error

# 定义RMSE
def calculate_rmse(targets, outputs):
    mse = mean_squared_error(targets, outputs)
    rmse = np.sqrt(mse)
    return rmse

# 定义MRE
def calculate_mre(targets, outputs):
    relative_errors = np.abs((targets - outputs) / targets)
    mre = np.mean(relative_errors)
    return mre

