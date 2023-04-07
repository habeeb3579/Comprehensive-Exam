#import numpy as np
#from plsonecomp import plsonecomp

import numpy as np
#import plsonecomp

#from packages import np, plsonecomp

def sub_pls(X, Y, lv):
    """
    pls calculates a PLS model
    
    Inputs:
    X: independent variables
    Y: dependent variable(s)
    lv: number of latent variables
    
    Outputs:
    P: loadings for X
    Q: loadings for Y
    W: weights for X
    T: scores for X
    U: scores for Y
    bsco: regression coefficients
    ssqdif: differences in sum of squares
    """
    import plsonecomp
    nX, mX = X.shape
    nY, mY = Y.shape
    
    if mX < lv:
        raise ValueError('The number of X variables is lower than the selected number of components')
    
    if nX != nY:
        raise ValueError('The number of X samples does not match the number of Y ditto')
    
    P = np.zeros((mX, lv))
    Q = np.zeros((mY, lv))
    W = np.zeros((mX, lv))
    T = np.zeros((nX, lv))
    U = np.zeros((nY, lv))
    bsco = np.zeros((1,lv))
    ssq = np.zeros((lv, 2))
    ssqX = np.sum(np.sum(X**2))
    ssqY = np.sum(np.sum(Y**2))
    #print(f'X is {X}')
    #print(f'Y is {Y}')
    #print(f'ssqX is {ssqX}')
    #print(f'ssqY is {ssqY}')
    
    for i in range(lv):
        p, q, w, t, u = plsonecomp.plsonecomp(X, Y) # Subfunction
        #if i==0:
        #    print(f'p is {p}')
        #    print(f'q is {q}')
        #    print(f'w is {w}')
        #    print(f't is {t}')
        #    print(f'u is {u}')
        bsco[:,i] = u.T @ t / (t.T @ t)
        #print(f'bsco {i} is {bsco.shape}')
        #print(f'Y {i} is {Y.shape}')
        #print(f't {i} is {t.shape}')
        #print(f'p {i} is {q.T.shape}')
        X = X - (t.reshape(-1,1)@p.reshape(-1,1).T)
        if np.max(q.shape)==1:
            Y = Y - bsco[:,i] * (t.reshape(-1,1)*q.reshape(-1,1)[:,0])
        else:
            Y = Y - bsco[:,i] * (t.reshape(-1,1)@q.reshape(-1,1).T)
        ssq[i, 0] = np.sum(np.sum(X**2)) * 100 / ssqX
        ssq[i, 1] = np.sum(np.sum(Y**2)) * 100 / ssqY
        T[:, i] = t #[:, 0]
        U[:, i] = u #[:, 0]
        P[:, i] = p #[:, 0]
        W[:, i] = w #[:, 0]
        Q[:, i] = q #[:, 0]
    
    ssqdif = np.zeros((lv, 2))
    ssqdif[0, 0] = 100 - ssq[0, 0]
    ssqdif[0, 1] = 100 - ssq[0, 1]
    
    for i in range(1, lv):
        for j in range(2):
            ssqdif[i, j] = -ssq[i, j] + ssq[i-1, j]
    
    return P, Q, W, T, U, bsco, ssqdif