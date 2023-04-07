import numpy as np
from sys import stdout
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter

def band_selection_sa(X,y,n_of_bands, max_lv, n_iter):
 
    p = np.arange(X.shape[1])
    np.random.shuffle(p)
    bands = p[:n_of_bands] # Selected Bands. Start off with a random selection
    nbands = p[n_of_bands:] # Excluded bands
 
    Xop = X[:,bands] #This is the array to be optimised
 
    # Run a PLS optimising the number of latent variables
    opt_comp, rmsecv_min = pls_optimise_components(Xop, y, max_lv)
 
    rms = [] # Here we store the RMSE value as the optimisation progresses     
    for i in range(n_iter):
 
        cool = 0.001*rmsecv_min # cooling parameter. It decreases with the RMSE
        new_bands = np.copy(bands)
        new_nbands = np.copy(nbands)
        
        
        #swap three elements at random
        el1 = np.random.choice(new_bands, size=3, replace=False)
        r1 = [np.where(new_bands==i)[0][0] for i in el1]
        
        el2 = np.random.choice(new_nbands, size=3, replace=False)
        r2 = [np.where(new_nbands==i)[0][0] for i in el2]
        
        new_bands[r1] = el2
        new_nbands[r2] = el1
 
 
        Xop = X[:,new_bands]
 
        opt_comp_new, rmsecv_min_new = pls_optimise_components(Xop, y, max_lv)
 
        # If the new RMSE is less than the previous, accept the change
        if (rmsecv_min_new < rmsecv_min): 
            bands = new_bands 
            nbands = new_nbands 
            opt_comp = opt_comp_new 
            rmsecv_min = rmsecv_min_new 
            rms.append(rmsecv_min_new) 
 
            stdout.write("\r"+str(i)) 
            stdout.write(" ") 
            stdout.write(str(opt_comp_new)) 
            stdout.write(" ") 
            stdout.write(str(rmsecv_min_new)) 
            stdout.flush() 
 
        # If the new RMSE is larger than the previous, accept it with some probability # dictated by the cooling parameter 
        if (rmsecv_min_new > rmsecv_min):
            
            prob = np.exp(-(rmsecv_min_new - rmsecv_min)/cool) # probability
            if (np.random.random() < prob):
                bands = new_bands
                nbands = new_nbands
                opt_comp = opt_comp_new
                rmsecv_min = rmsecv_min_new
                rms.append(rmsecv_min_new)
                
                stdout.write("\r"+str(i))
                stdout.write(" ")
                stdout.write(str(opt_comp_new))
                stdout.write(" ")
                stdout.write(str(rmsecv_min_new))
                stdout.flush()
            else:
                rms.append(rmsecv_min)
    
    stdout.write("\n")
    print(np.sort(bands))
    print('end')
 
    return(bands, opt_comp,rms)

def base_pls(X,y,n_components, return_model=False):
 
    # Simple PLS
    pls_simple = PLSRegression(n_components=n_components)
    # Fit
    pls_simple.fit(X, y)
    # Cross-validation
    y_cv = cross_val_predict(pls_simple, X, y, cv=10)
 
    # Calculate scores
    score = r2_score(y, y_cv)
    rmsecv = np.sqrt(mean_squared_error(y, y_cv))
 
    if return_model == False:
        return(y_cv, score, rmsecv)
    else:
        return(y_cv, score, rmsecv, pls_simple)
    

def pls_optimise_components(X, y, npc):
 
    rmsecv = np.zeros(npc)
    for i in range(1,npc+1,1):
 
        # Simple PLS
        pls_simple = PLSRegression(n_components=i)
        # Fit
        pls_simple.fit(X, y)
        # Cross-validation
        y_cv = cross_val_predict(pls_simple, X, y, cv=10)
 
        # Calculate scores
        score = r2_score(y, y_cv)
        rmsecv[i-1] = np.sqrt(mean_squared_error(y, y_cv))
 
    # Find the minimum of ther RMSE and its location
    opt_comp, rmsecv_min = np.argmin(rmsecv),  rmsecv[np.argmin(rmsecv)]
 
    return (opt_comp+1, rmsecv_min)

def regression_plot(y_ref, y_pred, title = None, variable = None):
 
    # Regression plot
    z = np.polyfit(y_ref, y_pred, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_ref, y_pred, c='red', edgecolors='k')
        ax.plot(y_ref, z[1]+z[0]*y_ref, c='blue', linewidth=1)
        ax.plot(y_ref, y_ref, color='green', linewidth=1)
 
        if title is not None:
            plt.title(title)
        if variable is not None:
            plt.xlabel('Measured ' + variable)
            plt.ylabel('Predicted ' + variable)
 
        plt.show()