import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as collections
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

from sys import stdout
#from packages import np, pd, plt, collections, PLSRegression, cross_val_predict, mean_squared_error, r2_score, stdout
def pls_variable_selection1(X, y, max_comp):
    
    # Define MSE array to be populated
    rmse = np.zeros((max_comp,X.shape[1]))
 
    # Loop over the number of PLS components
    for i in range(max_comp):
        
        # Regression with specified number of components, using full spectrum
        pls1 = PLSRegression(n_components=i+1)
        pls1.fit(X, y)
        
        # Indices of sort spectra according to ascending absolute value of PLS coefficients
        sorted_ind = np.argsort(np.abs(pls1.coef_[:,0]))
 
        # Sort spectra accordingly 
        Xc = X[:,sorted_ind]
        
        samp, feat = Xc.shape
 
        # Discard one wavelength at a time of the sorted spectra,
        # regress, and calculate the MSE cross-validation
        #there has to be a minimum of i+1 features in order to match the n_comp
        for j in range(feat-(i+1)):
 
            pls2 = PLSRegression(n_components=i+1)
            pls2.fit(Xc[:, j:], y)
            
            y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv=10)
 
            rmse[i,j] = np.sqrt(mean_squared_error(y, y_cv))
    
        comp = 100*(i+1)/(max_comp)
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
 
    # # Calculate and print the position of minimum in MSE
    rmseminx,rmseminy = np.where(rmse==np.min(rmse[np.nonzero(rmse)]))
    #mseminx, mseminy = np.unravel_index(mse[mse > 0].argmin(), mse.shape)
 
    print("Optimised number of PLS components: ", rmseminx[0]+1)
    print("Wavelengths to be discarded ",rmseminy[0])
    print('Optimised MSEP ', rmse[rmseminx,rmseminy][0])
    stdout.write("\n")
    # plt.imshow(mse, interpolation=None)
    # plt.show()
 
 
    # Calculate PLS with optimal components and export values
    pls = PLSRegression(n_components=rmseminx[0]+1)
    pls.fit(X, y)
        
    sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))
 
    Xc = X[:,sorted_ind]
 
    return(Xc, Xc[:,rmseminy[0]:],rmseminx[0]+1,rmseminy[0], sorted_ind, sorted_ind[rmseminy[0]:])