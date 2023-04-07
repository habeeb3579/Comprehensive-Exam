from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
import numpy as np

def pls_coeff_sel(X, y, sel_wavelengths=50):
    """
    Variable selection using pls coefficient values.

    :param x: numpy array of shape (n_samples, n_features) containing the spectral data
    :param y: numpy array of shape (n_samples, n_targets) containing the target data
    :param sel_wavelengths: number of wavelengths to be selected
    :return: a tuple containing the selected wavelength, mse, and components used
    """
    
    
    params = {'n_components':list(range(1,20))}

    kf2 = KFold(n_splits=5, random_state=32, shuffle=True)
    plsr = PLSRegression()

    model_plsr1 = GridSearchCV(plsr, params, cv=kf2)
    model_plsr1.fit(X,y)
    comp = model_plsr1.best_params_['n_components']

    #pls = PLSRegression(n_components=num_selected_wavelengths)
    #pls.fit(X, y)

    coeffsidx = np.argsort(np.abs(model_plsr1.best_estimator_.coef_), axis=0).flatten()[::-1]
    selected_wavelengths = coeffsidx[:sel_wavelengths]
    
    y_cv = cross_val_predict(model_plsr1, X[:,selected_wavelengths], y, cv=5)
    mse = mean_squared_error(y, y_cv)

    return selected_wavelengths, mse, comp

  