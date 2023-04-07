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

def partial_least_squares(X, y, num_selected_wavelengths=10):
    # Preprocess the data by standardizing it
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y[:, np.newaxis])

    pls = PLSRegression(n_components=num_selected_wavelengths)
    pls.fit(X, y)

    loadings = np.abs(pls.x_loadings_[:, 0:num_selected_wavelengths].sum(axis=1))
    selected_wavelengths = np.argsort(loadings)[::-1][:num_selected_wavelengths]

    return selected_wavelengths

def simultaneous_idc_pls(X, y, num_selected_wavelengths=10):
    selected_wavelengths_idc = iterative_distance_correlation(X, y, num_selected_wavelengths)
    selected_wavelengths_pls = partial_least_squares(X, y, num_selected_wavelengths)
    
    # Combine the selected wavelengths from IDC and PLS
    selected_wavelengths = np.intersect1d(selected_wavelengths_idc, selected_wavelengths_pls)
    
    return selected_wavelengths

# Load the spectral data and response into numpy arrays X and y
#X = np.loadtxt('spectral_data.txt')
#y = np.loadtxt('response.txt')

# Select the wavelengths using IDC and PLS
#selected_wavelengths = simultaneous_idc_pls(X, y, num_selected_wavelengths=10)

# Use the selected wavelengths to create a new dataset
#X_selected = X[:, selected_wavelengths]