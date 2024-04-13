import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from timeit import default_timer as timer
from task1 import compute_map, compute_ndcg


# ParameterGrid
param_grid = {
    'learning_rate': [0.1, 1],
    'alpha': [0, 0.1],
    'gamma': [0, 0.1, 1],
    'max_depth': [6, 7],
    'n_estimators': [100, 200]
}

best_ndcg = 0
best_map = 0
best_params_index = 0
parameter_grid = ParameterGrid(param_grid)

for i, params in enumerate(parameter_grid):
    print(params)