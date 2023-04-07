import numpy as np
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
    #print(f'AllRMSE {AllRMSE}')
    RedRMSE = AllRMSE[:,1:] # PLSC0 is excluded in finding the first local minimum
    SignMat = np.hstack([np.sign(np.diff(RedRMSE, axis=1)), np.ones((Model["intervals"], 1))])
    #print(f'sign mat {RedRMSE.shape} and {SignMat}')
    minRMSEinInterval = []
    for i in range(RedRMSE.shape[0]):
        for j in range(RedRMSE.shape[1]):
            if SignMat[i,j] == 1:
                minRMSEinInterval.append(RedRMSE[i,j])
                break
    #minRMSEinInterval = np.flipud(minRMSEinInterval)
    minRMSEinInterval = np.array(minRMSEinInterval)
    #print(f'MinRMSEint {minRMSEinInterval}')
    minRMSEinInterval = minRMSEinInterval.T #reshape(-1,1) #T

    ix_sorted = np.argsort(minRMSEinInterval)
    RMSEsorted = np.flipud(np.sort(minRMSEinInterval))
    ix_sorted = np.flipud(ix_sorted)
    #print(f'ix_sorted {ix_sorted}')
    #print(f'RMSEsorted {RMSEsorted}')
    l = len(ix_sorted)
    #print(f'l is {l}')
    ix_for_iterative = ix_sorted[l-1]
    #print(f'ix iterative is {ix_for_iterative}')
    RMSEmin = RMSEsorted[l-1]
    #print(f'RMSEmin is {RMSEmin}')

    RedRMSEglobal = Model["PLSmodel"][Model["intervals"]]["RMSE"][1:] # PLSC0 is excluded in finding the first local minimum
    #print(f'rmse global {RedRMSEglobal}')
    SignMat = np.hstack([np.sign(np.diff(RedRMSEglobal)), 1])
    for j in range(RedRMSEglobal.shape[0]):
        if SignMat[j] == 1:
            minRMSEglobal = RedRMSEglobal[j]
            break
            
    #print(f'min rmse global {minRMSEglobal}')
    
    return RMSEsorted, ix_sorted, RMSEmin, ix_for_iterative, minRMSEglobal 