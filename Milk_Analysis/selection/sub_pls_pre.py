import numpy as np

def sub_pls_pre(Xpred, bsco, P, Q, W, lv):
    nX, mX = Xpred.shape
    nQ, mQ = Q.shape
    t_hat = np.zeros((nX, lv))
    Ypred = np.zeros((nX, nQ, lv))
    for i in range(lv):
        t_hat[:, i] = (Xpred@W[:, i])
        Xpred = Xpred - (t_hat[:, i].reshape(-1,1)@P[:, i].reshape(-1,1).T)
    temp = 0
    for i in range(lv):
        temp = temp + bsco[0, i] * (t_hat[:, i].reshape(-1,1)@Q[:, i].reshape(-1,1).T)
        Ypred[:, :, i] = temp
    return Ypred