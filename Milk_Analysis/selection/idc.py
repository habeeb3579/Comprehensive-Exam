import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

def iterative_distance_correlation(X, y, num_selected_wavelengths=10):
    # Preprocess the data by standardizing it
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y[:, np.newaxis])

    selected_wavelengths = []
    while len(selected_wavelengths) < num_selected_wavelengths:
        max_corr = 0
        max_i = -1
        for i in range(X.shape[1]):
            if i in selected_wavelengths:
                continue

            corr = np.abs(cdist(X[:, i][:, np.newaxis], y, metric='correlation')).squeeze()
            if corr > max_corr:
                max_corr = corr
                max_i = i

        if max_i == -1:
            break
        
        selected_wavelengths.append(max_i)
    
    return selected_wavelengths