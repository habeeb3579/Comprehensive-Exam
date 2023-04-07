import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as collections
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

from sys import stdout

#from packages import np, pd, plt, collections, PLSRegression, cross_val_predict, mean_squared_error, r2_score, stdout
def mwpls(X,y, win_size, plot=False):
    win_pos = np.arange(0, X.shape[1]-win_size)
    #min(20,win_size) tracks latent variable components
    rmse = np.zeros((X.shape[1]-win_size,min(20,win_size)))
    
    for i in win_pos:
        for j in range(min(20,win_size)): # number of PLS components is limited by the win size
            pls = PLSRegression(n_components=j+1)
            # Fit
            pls.fit(X[:,i:i+win_size], y)
            # Cross-validation
            y_cv = cross_val_predict(pls, X[:,i:i+win_size], y, cv=10)
 
            rmse[i,j] = np.sqrt(mean_squared_error(y_cv, y))
 
        comp = 100*(i+1)/(X.shape[1]-win_size)
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
 
    stdout.write("\n")
 
    # Calculate and print the position of minimum in MSE
    #rminx is win start, rminy is ncomp
    rmseminx,rmseminy = np.where(rmse==np.min(rmse[np.nonzero(rmse)]))
    print("Suggested window position: ",rmseminx[0],rmseminy[0], rmse[rmseminx,rmseminy])
    
    if plot is True:
        plt.imshow(rmse, interpolation=None, aspect=0.1)
        plt.colorbar()
        plt.show()
    
    return(rmseminx[0],rmseminy[0], rmse[rmseminx,rmseminy]) 


#rmseminy = 16
#rmseminx = 104
#win_size = 80
#pls = PLSRegression(n_components=rmseminy+1)
# Fit
#pls.fit(X[:,rmseminx:rmseminx+win_size], y)
# Cross-validation
#y_cv = cross_val_predict(pls, X[:,rmseminx:rmseminx+win_size], y, cv=10)
# Calculate scores
#score_mw = r2_score(y, y_cv)
#rmse_mw = np.sqrt(mean_squared_error(y, y_cv))
 
# Fit a line to the CV vs response
#z = np.polyfit(y, y_cv, 1)
#with plt.style.context(('ggplot')):
#    fig, ax = plt.subplots(figsize=(9, 5))
#    ax.scatter(y_cv, y, c='red', edgecolors='k')
    #Plot the best fit line
#    ax.plot(np.polyval(z,y), y, c='blue', linewidth=1)
    #Plot the ideal 1:1 line
#    ax.plot(y, y, color='green', linewidth=1)
 
#    plt.title('$R^{2}$ (CV): %5.2f ' % score_mw) 
#    plt.xlabel('Predicted value')
#    plt.ylabel('Measured value')
#    plt.show()