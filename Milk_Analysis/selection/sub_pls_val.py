import numpy as np
from preprop import rmbi, scalenew, auto, mncn, msc, msc_pre, scaleback
#from sub_iplsreverse import sizer
#from sub_pls import sub_pls
#from sub_pls_pre import sub_pls_pre

#import sub_iplsreverse #import sizer
#import sub_pls
#import sub_pls_pre


#from packages import sizer, sub_pls_pre, sub_pls
def sub_pls_val(X, Y, no_of_lv, prepro_method, val_method, segments):
    """
    sub_pls_val for PLS modelling with selected validation method

    Input:
    X contains the independent variables
    Y contains the dependent variable(s), NOTE: Y is allways autoscaled
    no_of_lv is the number of PLS components
    prepro_method is 'mean', 'auto', 'mscmean', 'mscauto' or 'none'
    val_method is 'test', 'full', 'syst111', 'syst123', 'random' or 'manual'
    segments is number of segments in cross validation
        if val_method is 'test' then segments should be a column vector with test set indices
        if val_method is 'manual' then segments should be a cell array, see makeManualSegments
    Output:
    PLSmodel is a structured array containing all model and validation information

    Subfunctions at the end of this file: rmbi, msc, msc_pre, mncn, auto

    Lars Nørgaard, July 2004

    PLSmodel = sub_pls_val(X,Y,no_of_lv,prepro_method,val_method,segments);
    """
    import sub_iplsreverse #import sizer
    import sub_pls
    import sub_pls_pre
    n, m = X.shape
    o, p = Y.shape
    #print(f'p is {p}')

    if np.max(sub_iplsreverse.sizer(segments)) == 1 and isinstance(segments, int):  # Not test or manual
        no_sampl = n // segments
        left_over_samples = n % segments
        #print(f'n of samp {no_sampl} and left over {left_over_samples}')

    if isinstance(segments, (list, np.ndarray)):
        manualseg = segments  # A list of lists
        segments = np.max(segments.shape)  # Now a scalar
        Nsamples = 0
        for j in range(segments):
            Nsamples += np.max(sub_iplsreverse.sizer(manualseg[j]))
        if n != Nsamples:
            print('The number of samples in X does not correspond to the number of samples in manualseg')
            return

    Ypred = np.zeros((n, p, no_of_lv))
    Ypred0 = np.zeros((n,p))
    count = 1

    PLSmodel = {'prepro_method': prepro_method, 'val_method': val_method, 'cv':[]}

    if val_method == 'full':
        val_method = 'syst123'
        segments = n

    if val_method == 'random':
        ix = np.random.rand(n, 1)
        ix = np.argsort(ix.flatten())

    

    if val_method == 'test':
        tot = np.arange(0, n)
        tot = np.delete(tot, segments-1, axis=0)
        cal = tot
        Xseg = X[cal, :]
        Yseg = Y[cal, :]
        Yseg, my, stdy = auto(Yseg)
        if prepro_method.lower() == 'mean':
            Xseg, mx = mncn(Xseg)
            Xpseg = scalenew(X[segments-1, :], mx)
        elif prepro_method.lower() == 'auto':
            Xseg, mx, stdx = auto(Xseg)
            Xpseg = scalenew(X[segments-1, :], mx, stdx)
        elif prepro_method.lower() == 'mscmean':
            Xseg, Xsegmeancal = msc(Xseg)
            Xseg, mx = mncn(Xseg)
            Xpseg = msc_pre(X[segments-1, :], Xsegmeancal)
            Xpseg = scalenew(Xpseg, mx)
        elif prepro_method.lower() == 'mscauto':
            Xseg, Xsegmeancal = msc(Xseg)
            Xseg, mx, stdx = auto(Xseg)
            Xpseg = msc_pre(X[segments-1, :], Xsegmeancal)
            Xpseg = scalenew(Xpseg, mx, stdx)
        elif prepro_method.lower() == 'none':
            pass
        P, Q, W, T, U, bsco, ssqdif = sub_pls.sub_pls(Xseg, Yseg, no_of_lv)
        Ypred[segments-1, :, :] = sub_pls_pre.sub_pls_pre(Xpseg, bsco, P, Q, W, no_of_lv)
        for j in range(no_of_lv):
            Ypred[segments-1, :, j] = scaleback(Ypred[segments-1, :, j], my, stdy)
        Ypred0[segments-1, :] = np.ones((np.max(sub_iplsreverse.sizer(segments)),1))*np.zeros(sub_iplsreverse.sizer(my))
        Ypred0[segments-1, :] = scaleback(Ypred0[segments-1, :], my, stdy)
        PLSmodel['test'] = segments
    elif val_method in ['full', 'syst111', 'syst123', 'random', 'manual']:
        for i in range(segments):
            if val_method == 'syst111':
                if left_over_samples == 0:
                    count = count
                    p_cvs = np.arange((i-1+1)*no_sampl+1+(count-1)-1, (i+1)*no_sampl+(count-1))
                else:
                    p_cvs = np.arange((i-1+1)*no_sampl+1+(count-1)-1, (i+1)*no_sampl+count)
                    count += 1
                    left_over_samples -= 1
                if i==0 or i==1 or i==2:
                    print(f'pcvs as {p_cvs}')
            elif val_method == 'syst123':
                p_cvs = np.arange(i, n, segments)
                #print(f'pcvs is {p_cvs}')
            elif val_method == 'random':
                p_cvs = np.arange(i, n, segments)
                p_cvs = ix[p_cvs]
            elif val_method == 'manual':
                p_cvs = manualseg[i]
            tot = np.arange(0, n)
            tot = np.delete(tot, p_cvs, axis=0)
            m_cvs = tot
            #print(f'mcvs is {m_cvs}')
            #PLSmodel['cv'][i] = p_cvs
            PLSmodel['cv'].append(p_cvs)
            Xseg = X[m_cvs,:]
            Yseg = Y[m_cvs,:]
            Xpseg = X[p_cvs,:]
            #print(f'Yseg is {Yseg}')
            Yseg, my, stdy = auto(Yseg)
            #print(f'yseg new is {Yseg}')
            prepro_method_lower = prepro_method.lower()
            if prepro_method_lower == 'mean':
                Xseg, mx = mncn(Xseg)
                Xpseg = scalenew(Xpseg, mx)
            elif prepro_method_lower == 'auto':
                Xseg, mx, stdx = auto(Xseg)
                Xpseg = scalenew(Xpseg, mx, stdx)
            elif prepro_method_lower == 'mscmean':
                Xseg, Xsegmeancal = msc(Xseg)
                Xseg, mx = mncn(Xseg)
                Xpseg = msc_pre(Xpseg, Xsegmeancal)
                Xpseg = scalenew(Xpseg, mx)
            elif prepro_method_lower == 'mscauto':
                Xseg, Xsegmeancal = msc(Xseg)
                Xseg, mx, stdx = auto(Xseg)
                Xpseg = msc_pre(Xpseg, Xsegmeancal)
                Xpseg = scalenew(Xpseg, mx, stdx)
            elif prepro_method_lower == 'none':
                pass

            P, Q, W, T, U, bsco, ssqdif = sub_pls.sub_pls(Xseg, Yseg, no_of_lv)
            #print(f'P is {P}')
            #print(f'Q is {Q}')
            #print(f'W is {W}')
            #print(f'T is {T}')
            #print(f'U is {U}')
            #print(f'bsco is {bsco}')
            #print(f'ssqdif is {ssqdif}')
            Ypred[p_cvs,:,:] = sub_pls_pre.sub_pls_pre(Xpseg, bsco, P, Q, W, no_of_lv)
            #print(f'ypredpcvs is {Ypred[p_cvs,:,:]}')

            for j in range(no_of_lv):
                Ypred[p_cvs,:,j] = scaleback(Ypred[p_cvs,:,j], my, stdy)
            #print(f'ypredo {my}')
            Ypred0[p_cvs,:] = np.ones((np.max(sub_iplsreverse.sizer(p_cvs)),1))*np.zeros(sub_iplsreverse.sizer(my))
            Ypred0[p_cvs,:] = scaleback(Ypred0[p_cvs,:], my, stdy)
            


    #print(f'ypredo {Ypred0}')
    if val_method == 'test':
        RMSE0, Bias0 = rmbi(Y[segments-1, :], Ypred0[segments-1, :])
    else:
        RMSE0, Bias0 = rmbi(Y, Ypred0)

    RMSE, Bias = np.zeros(no_of_lv), np.zeros(no_of_lv)
    for i in range(no_of_lv):
        if val_method == 'test':
            RMSE[i], Bias[i] = rmbi(Y[segments-1, :], Ypred[segments-1, :, i])
        else:
            RMSE[i], Bias[i] = rmbi(Y, Ypred[:, :, i])

    PLSmodel = {}
    PLSmodel['Ypred0'] = Ypred0
    PLSmodel['Ypred'] = Ypred
    PLSmodel['RMSE'] = np.hstack((RMSE0, RMSE))
    PLSmodel['Bias'] = np.hstack((Bias0, Bias))

    # Global model with all samples
    Y, my, stdy = auto(Y)
    if prepro_method.lower() == 'mean':
        X, mx = mncn(X)
    elif prepro_method.lower() == 'auto':
        X, mx, stdx = auto(X)
    elif prepro_method.lower() == 'mscmean':
        X = msc(X)
        X, mx = mncn(X)
    elif prepro_method.lower() == 'mscauto':
        X = msc(X)
        X, mx, stdx = auto(X)
    elif prepro_method.lower() == 'none':
        print('No scaling')
        # To secure that no centering/scaling is OK    

    PLSmodel['P'], PLSmodel['Q'], PLSmodel['W'], PLSmodel['T'], PLSmodel['U'], PLSmodel['bsco'], PLSmodel['ssqdif'] = sub_pls.sub_pls(X, Y, no_of_lv)
    return PLSmodel