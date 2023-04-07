import numpy as np

def plsonecomp(X, Y):
    ny, my = Y.shape
    if my == 1:
        u = Y[:,0]
    else:
        SumSquaresY = np.sum(Y**2, axis=0)
        yi = np.argmax(SumSquaresY)
        u = Y[:,yi]
    t_difference = 100
    t_old = X[:,0]
    it_count = 1
    #print(f'u is {u}')
    # Conversion limit: 1e-10
    while t_difference > 1e-10:
        it_count += 1
        w = (u.T@X).T
        w = (w.T / np.linalg.norm(w.T)).T
        t = X@w
        if my == 1:
            q = np.array([1])
            break
        q = (t.T@ Y).T
        q = (q.T / np.linalg.norm(q.T)).T
        u = Y@q
        t_difference = np.linalg.norm(t_old - t)
        t_old = t
        if it_count >= 1000:
            print('No convergence up to 1000 iterations')
            break
    p = ((t.T@X) / (t.T@t)).T
    #print(f'p is {p}')
    p_norm = np.linalg.norm(p)
    #print(f'pnorm is {p_norm}')
    t = t * p_norm
    #print(f't is {t}')
    w = w * p_norm
    p = p / p_norm
    return p, q, w, t, u