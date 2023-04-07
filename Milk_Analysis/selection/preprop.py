import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import FLOAT_DTYPES
import scipy
from scipy import newaxis as nA
# Subfunctions rmbi, msc, msc_pre, mncn, auto, scalenew, scaleback
def rmbi(Yref, Ypred):
    '''RMSE and Bias calculation'''
    n, m = Yref.shape
    RMSE = np.sqrt(np.sum(np.sum(np.square(Ypred - Yref))) / (n * m))
    Bias = np.sum(np.sum(Ypred - Yref))/(n*m)
    #np.sum(Ypred - Yref) / (n * m)
    return RMSE, Bias


def snv(input_data):
  
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
 
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
 
    return output_data


def msc(X):
    ''' Perform Multiplicative scatter correction without a reference spectrum'''
    n, m = X.shape
    Xmeancal = np.mean(X, axis=0)
    Xmsc = np.zeros((n, m))
    for i in range(n):
        coef = np.polyfit(Xmeancal, X[i, :], 1)
        Xmsc[i, :] = (X[i, :] - coef[1]) / coef[0]
    return Xmsc, Xmeancal

def msc_pre(Xp, Xmeancal):
    ''' Perform Multiplicative scatter correction with a reference spectrum'''
    n, m = Xp.shape
    Xpmsc = np.zeros((n, m))
    for i in range(n):
        coef = np.polyfit(Xmeancal, Xp[i, :], 1)
        Xpmsc[i, :] = (Xp[i, :] - coef[1]) / coef[0]
    return Xpmsc

def mncn(X):
    '''Mean centering'''
    n, m = X.shape
    meanX = np.mean(X, axis=0)
    Xmean = X - meanX
    return Xmean, meanX

def auto(X):
    ''' autosclaing'''
    n, m = X.shape
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0)
    Xauto = (X - meanX) / stdX
    return Xauto, meanX, stdX

def scalenew(Xnew, meanXold, stdXold=None):
    """resclaing"""
    n, m = Xnew.shape
    if stdXold is None:
        Xscalenew = Xnew - meanXold
    else:
        Xscalenew = (Xnew - meanXold) / stdXold
    return Xscalenew

def scaleback(X, meanX, stdX=None):
    """scale back"""
    n, m = X.shape
    if stdX is None:
        Xscaleback = X + meanX
    else:
        Xscaleback = (X * stdX) + meanX
    return Xscaleback

class SavitzkyGolay(TransformerMixin, BaseEstimator):

    def __init__(self, *, filter_win=31, poly_order=1, deriv_order=0, delta=1.0, copy=True):
        self.copy = copy
        self.filter_win = filter_win
        self.poly_order = poly_order
        self.deriv_order = deriv_order
        self.delta = delta

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        # Make sure filter window length is odd
        filter_win = self.filter_win
        if self.filter_win % 2 == 0:
            filter_win += 1

        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=True, accept_sparse='csr', copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        X = savgol(X.T, filter_win=filter_win, poly_order=self.poly_order, deriv_order=self.deriv_order, delta=self.delta).T
        return X

    def _more_tags(self):
        return {'allow_nan': True}
    


def savgol(spectra, filter_win=31, poly_order=1, deriv_order=0, delta=1.0):
    """ Perform Savitzky–Golay filtering on the data (also calculates derivatives). This function is a wrapper for
    scipy.signal.savgol_filter.
    Args:
        spectra <numpy.ndarray>: NIRS data matrix.
        filter_win <int>: Size of the filter window in samples (default 11).
        poly_order <int>: Order of the polynomial estimation (default 3).
        deriv_order <int>: Order of the derivation (default 0).
    Returns:
        spectra <numpy.ndarray>: NIRS data smoothed with Savitzky-Golay filtering
    """
    return scipy.signal.savgol_filter(spectra, filter_win, poly_order, deriv_order, delta=delta, axis=0)



class GlobalStandardScaler(object):
    """Scales to unit standard deviation and mean centering using global mean and std of X, skleran like API"""
    def __init__(self,with_mean=True, with_std=True, normfact=1.0):
        self._with_mean = with_mean
        self._with_std = with_std
        self.std = None
        self.normfact=normfact
        self.mean = None
        self._fitted = False
        
    def fit(self,X, y = None):
        X = np.array(X)
        self.mean = X.mean()
        self.std = X.std()
        self._fitted = True
        
    def transform(self,X, y=None):
        if self._fitted:
            X = np.array(X)
            if self._with_mean:
                X=X-self.mean
            if self._with_std:
                X=X/(self.std*self.normfact)
            return X
        else:
            print("Scaler is not fitted")
            return
            
    def inverse_transform(self,X, y=None):
        if self._fitted:
            X = np.array(X)
            if self._with_std:
                X=X*self.std*self.normfact
            if self._with_mean:
                X=X+self.mean
            return X
        else:
            print("Scaler is not fitted")
            return
            
    def fit_transform(self,X, y=None):
        self.fit(X)
        return self.transform(X)



class EmscScaler(object):
    def __init__(self,order=1):
        self.order = order
        self._mx = None
        
    def mlr(self,x,y):
        """Multiple linear regression fit of the columns of matrix x 
        (dependent variables) to constituent vector y (independent variables)
        
        order -     order of a smoothing polynomial, which can be included 
                    in the set of independent variables. If order is
                    not specified, no background will be included.
        b -         fit coeffs
        f -         fit result (m x 1 column vector)
        r -         residual   (m x 1 column vector)
        """
        
        if self.order > 0:
            s=scipy.ones((len(y),1))
            for j in range(self.order):
                s=scipy.concatenate((s,(scipy.arange(0,1+(1.0/(len(y)-1)),1.0/(len(y)-1))**j)[:,nA]),1)
                #s=scipy.concatenate((s,((scipy.arange(0,1+(1.0/(len(y)-1)),1.0/(len(y)-1))**j)[:,nA])[:(len(s)),:]),1)
            X=scipy.concatenate((x, s),1)
        else:
            X = x
        
        #calc fit b=fit coefficients
        b = scipy.dot(scipy.dot(scipy.linalg.pinv(scipy.dot(scipy.transpose(X),X)),scipy.transpose(X)),y)
        f = scipy.dot(X,b)
        r = y - f

        return b,f,r

    
    def inverse_transform(self, X, y=None):
        print("Warning: inverse transform not possible with Emsc")
        return X
    
    def fit(self, X, y=None):
        """fit to X (get average spectrum), y is a passthrough for pipeline compatibility"""
        self._mx = scipy.mean(X,axis=0)[:,nA]
        
    def transform(self, X, y=None, copy=None):
        if type(self._mx) == type(None):
            print("EMSC not fit yet. run .fit method on reference spectra")
        else:
            #do fitting
            corr = scipy.zeros(X.shape)
            for i in range(len(X)):
                b,f,r = self.mlr(self._mx, X[i,:][:,nA])
                corr[i,:] = scipy.reshape((r/b[0,0]) + self._mx, (corr.shape[1],))
            return corr
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
        
        

def dataaugment(x, betashift = 0.05, slopeshift = 0.05,multishift = 0.05):
    #Shift of baseline
    #calculate arrays
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    #Calculate relative position
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    #Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5

    #Multiplicative
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1

    x = multi*x + offset

    return x  