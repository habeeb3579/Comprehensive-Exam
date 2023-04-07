#from sub_iplsreverse import sub_iplsreverse
#import numpy as np

import numpy as np
#import sub_iplsreverse

#from packages import sub_iplsreverse, np
def sub_bipls_limit(X, Y, no_of_lv, prepro_method, intervals, MaxNoVars, xaxislabels, val_method, segments):
    import sub_iplsreverse

    if X==None:
        print(' ')
        print(' biplslimitModel=sub_bipls_limit(X,Y,no_of_lv,intervals,cv_method,prepro_method,MaxNoVars,xaxislabels,segments);')
        print(' ')
        print(' Example:')
        print(' biplslimitModel=sub_bipls_limit(X,Y,10,20,''syst123'',''mean'',400,[],5);')
        print(' ')
        return
    
    #ALSO IMPLEMENT WITH MANUAL CV AND TEST
    
    minRMSEwithout,ix_for_iterative,minRMSEglobal = np.zeros(intervals-1), np.zeros(intervals-1, dtype='int'), np.zeros(intervals-1)
    # Calculate results for each interval left out
    ModelReverse = sub_iplsreverse.sub_iplsreverse(X, Y, no_of_lv, prepro_method, intervals, xaxislabels, val_method, segments)
    temp1, temp2, minRMSEwithout[0], ix_for_iterative[0], minRMSEglobal[0] = sort_ipls(ModelReverse)
    newX = np.copy(X)
    #newX = np.delete(newX, np.s_[ModelReverse.allint[ix_for_iterative, 1]:ModelReverse.allint[ix_for_iterative, 2]], axis=1)
    dels = np.array(range(ModelReverse['allint'][ix_for_iterative[0], 1],ModelReverse['allint'][ix_for_iterative[0], 2]+1))
    newX = np.delete(newX, dels, axis=1)
    RevVars = [newX.shape[1]] #np.array([newX.shape[1]])
    #keeptrackofinterval = np.array(list(zip(range(1, intervals+1), range(1, intervals+1))))
    #RevIntInfo = keeptrackofinterval[ix_for_iterative[0], :].reshape(1, 2)
    #keeptrackofinterval = np.delete(keeptrackofinterval, ix_for_iterative[0], axis=0)
    #keeptrackofinterval[:, 0] = range(1, intervals)
    keeptrackofinterval = np.hstack((np.arange(0, intervals).reshape(-1, 1), np.arange(0, intervals).reshape(-1, 1)))
    RevIntInfo = keeptrackofinterval[ix_for_iterative[0],:]
    keeptrackofinterval = np.delete(keeptrackofinterval, ix_for_iterative[0], axis=0)
    keeptrackofinterval[:, 0] = np.arange(0, intervals-1)
    for i in range(1, intervals-1):
        RevVars.append(newX.shape[1])
        if RevVars[i] > MaxNoVars:
            ModelReverse = sub_iplsreverse.sub_iplsreverse(newX, Y, no_of_lv, prepro_method, intervals-(i-1+1), xaxislabels, val_method, segments)
            temp1,temp2,minRMSEwithout[i],ix_for_iterative[i],minRMSEglobal[i] = sort_ipls(ModelReverse)
            newX = np.delete(newX, np.array(range(ModelReverse['allint'][ix_for_iterative[i], 1],ModelReverse['allint'][ix_for_iterative[i], 2]+1)), axis=1)
            RevIntInfo = np.vstack((RevIntInfo, keeptrackofinterval[ix_for_iterative[i],:]))
            keeptrackofinterval = np.delete(keeptrackofinterval, ix_for_iterative[i], axis=0)
            keeptrackofinterval[:, 0] = np.arange(0, intervals-i-1)
        else:
            break
    
    #for i in range(1, intervals-1):
    #    RevVars = np.append(RevVars, newX.shape[1])
    #    if RevVars[i] > MaxNoVars:
    #        ModelReverse = sub_iplsreverse(newX, Y, no_of_lv, prepro_method, intervals-(i-1), xaxislabels, val_method, segments)
    #        temp1, temp2, minRMSEwithout[i], ix_for_iterative[i], minRMSEglobal[i] = sort_ipls(ModelReverse)
    #        newX = np.delete(newX, np.s_[ModelReverse.allint[ix_for_iterative[i], 1]:ModelReverse.allint[ix_for_iterative[i], 2]], axis=1)
    #        RevIntInfo = np.append(RevIntInfo, keeptrackofinterval[ix_for_iterative[i], :].reshape(1, 2), axis=0)
    #        keeptrackofinterval = np.delete(keeptrackofinterval, ix_for_iterative[i], axis=0)
    #        keeptrackofinterval[:, 0] = range(1, intervals-i)
    #    else:
    #        break
    #l = len(minRMSEwithout)
    #RevRMSE = np.append(minRMSEglobal.reshape(-1, 1), minRMSEwithout[l-1])
    #RevIntInfo = np.delete(RevIntInfo, 0, axis=1)
    #RevIntInfo = np.append(RevIntInfo, keeptrackofinterval[0, 1])
    #RevVars = np.append(RevVars, newX.shape[1])
    
    l = len(minRMSEwithout)
    RevRMSE = np.concatenate((minRMSEglobal.T, np.array([minRMSEwithout[l-1]])))
    RevIntInfo = np.delete(RevIntInfo, 0, axis=1)
    RevIntInfo = np.vstack((RevIntInfo, keeptrackofinterval[0,1]))
    RevVars.extend([newX.shape[1]])
    
    biplslimitModel = {'type': 'bipls_limit', 'rawX': X, 'rawY': Y, 'no_of_lv': no_of_lv, 'prepro_method': prepro_method, 
           'intervals': intervals, 'xaxislabels': xaxislabels,
           'val_method': val_method, 'segments': segments, 'RevIntInfo': RevIntInfo.flatten(), 'RevRMSE': RevRMSE,
           'RevVars': RevVars}

    #biplslimitModel = {}
    #biplslimitModel['type'] = 'bipls_limit'
    #biplslimitModel['rawX'] = X
    #biplslimitModel['rawY'] = Y
    #biplslimitModel['no_of_lv'] = no_of_lv
    #biplslimitModel['prepro_method'] = prepro_method
    #biplslimitModel['intervals'] = intervals
    #biplslimitModel['xaxislabels'] = xaxislabels
    #biplslimitModel['val_method'] = val_method
    #biplslimitModel['segments'] = segments
    #biplslimitModel['RevIntInfo'] = RevIntInfo
    #biplslimitModel['RevRMSE'] = RevRMSE
    #biplslimitModel['RevVars'] = RevVars
    
    return biplslimitModel


def sort_ipls(Model):
    """
    Sorts intervals from iplsreverse according to predictive ability.
    Input: Model (output from iplsreverse)
    Output:
    RMSEsorted: the first minimum RMSE value for each interval, sorted according to size
    ix_sorted: interval number, sorted according to first minimum RMSE
    """
    AllRMSE = []
    for i in range(Model["intervals"]):
        AllRMSE.append(Model["PLSmodel"][i]["RMSE"])
    AllRMSE = np.array(AllRMSE)
    RedRMSE = AllRMSE[:,1:] # PLSC0 is excluded in finding the first local minimum
    SignMat = np.hstack([np.sign(np.diff(RedRMSE, axis=1)), np.ones((Model["intervals"], 1))])
    minRMSEinInterval = []
    for i in range(RedRMSE.shape[0]):
        for j in range(RedRMSE.shape[1]):
            if SignMat[i,j] == 1:
                minRMSEinInterval.append(RedRMSE[i,j])
                break
    minRMSEinInterval = np.array(minRMSEinInterval)
    minRMSEinInterval = minRMSEinInterval.T #reshape(-1,1) #T

    ix_sorted = np.argsort(minRMSEinInterval)
    RMSEsorted = np.flipud(np.sort(minRMSEinInterval))
    ix_sorted = np.flipud(ix_sorted)
    l = len(ix_sorted)
    ix_for_iterative = ix_sorted[l-1]
    RMSEmin = RMSEsorted[l-1]

    RedRMSEglobal = Model["PLSmodel"][Model["intervals"]]["RMSE"][1:] # PLSC0 is excluded in finding the first local minimum
    SignMat = np.hstack([np.sign(np.diff(RedRMSEglobal)), 1])
    for j in range(RedRMSEglobal.shape[0]):
        if SignMat[j] == 1:
            minRMSEglobal = RedRMSEglobal[j]
            break
            
    
    return RMSEsorted, ix_sorted, RMSEmin, ix_for_iterative, minRMSEglobal  