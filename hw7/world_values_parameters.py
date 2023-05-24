import numpy as np

regression_knn_parameters = {
    'knn__n_neighbors': np.arange(1, 50),

    # Apply uniform weighting vs k for k Nearest Neighbors Regression
    ##### TODO(f): Change the weighting #####
    'knn__weights': ['distance']
    # 'knn__weights': ['uniform']
}