from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#from packages import np, PLSRegression, cross_val_predict, mean_squared_error, r2_score, plt

def mwplsr_wis(x, y, window_size, n_components, criterion='rmsep', plot=True):
    """
    Perform Moving Window Partial Least Squares Regression with Wavelength Interval Selection (MWPLSR-WIS) on the input data.

    :param x: numpy array of shape (n_samples, n_features) containing the spectral data
    :param y: numpy array of shape (targets, ) containing the target data
    :param window_size: the size of the moving window
    :param n_components: the number of PLS components to use
    :param criterion: the criterion to use for wavelength interval selection. Can be 'rmsecv' or 'rmsep'.
    :param plot: boolean to specify whether to generate 2D plot of the rmse or not 
    :return: a tuple containing the selected wavelength intervals and the corresponding PLS coefficients
    """
    n_samples, n_features = x.shape
    n_targets = 1

    rmse = np.zeros((n_features-window_size, 15))
    pls_coef = np.zeros((n_features, n_targets))
    wis = []
    for i in range(0, n_features-window_size):
        j = min(i + window_size, n_features)
        x_window = x[:, i:j]
        for k in range(min(15, j)):
            if criterion == 'rmsecv':
                pls = PLSRegression(n_components=k+1)
                # Use cross-validation to select the wavelength interval
                pls.fit(x_window, y)
                #rmsecv = np.mean(pls.cv_scores_['rmse'], axis=0)
                #best_interval = np.argmin(rmsecv)

                y_cv = cross_val_predict(pls, x[:,i:i+window_size], y, cv=10)
 
                rmse[i,k] = np.sqrt(mean_squared_error(y_cv, y))
            elif criterion == 'rmsep':
                # Use prediction error on a validation set to select the wavelength interval
                x_train, y_train, x_val, y_val = split_train_val(x_window, y)
                pls = PLSRegression(n_components=k+1)
                pls.fit(x_train, y_train)
                y_val_pred = pls.predict(x_val)
                #rmsep = np.sqrt(np.mean((y_val - y_val_pred)**2, axis=0))
                #best_interval = np.argmin(rmsep)

                #y_cv = cross_val_predict(pls, X[:,i:i+win_size], y, cv=10)
 
                rmse[i,k] = np.sqrt(mean_squared_error(y_val_pred, y_val))
            else:
                raise ValueError(f"Unknown criterion '{criterion}'")

        best_interval = np.argmin(rmse[i,:])
        # Store the selected wavelength interval and the corresponding PLS coefficients
        wis.append((i+best_interval, j-1))
        pls_coef[i:i+window_size, :] = pls.coef_

    win_start,opt_comp = np.where(rmse==np.min(rmse[np.nonzero(rmse)]))
    if plot is True:
        plt.imshow(rmse, interpolation=None, aspect=0.1)
        plt.colorbar()
        plt.show()
    return wis, pls_coef, rmse, win_start, opt_comp

def split_train_val(x, y, val_size=0.2):
    """
    Split the input data into training and validation sets.

    :param x: numpy array of shape (n_samples, n_features) containing the spectral data
    :param y: numpy array of shape (n_samples, n_targets) containing the target data
    :param val_size: the proportion of data to use for validation
    :return: a tuple containing the training and validation sets for x and y
    """
    n_samples = x.shape[0]
    val_start = int(n_samples * (1 - val_size))
    x_train = x[:val_start, :]
    y_train = y[:val_start, :]
    x_val = x[val_start:, :]
    y_val = y[val_start:, :]
    return x_train, y_train, x_val, y_val