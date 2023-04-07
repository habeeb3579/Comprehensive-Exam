#import numpy as np
#from sub_pls_val import sub_pls_val

import numpy as np
#import sub_pls_val


#from packages import np, sub_pls_val

def sipls(X,Y,no_of_lv,prepro_method,intervals,no_of_comb,xaxislabels,val_method,segments):
    '''
    sipls calculates PLS models on intervals combinations (2, 3, or 4)

    Input:
    X: independent data
    Y: dependent data
    no_of_lv: maximum number of PLS components
    prepro_method: 'none', 'mean', 'auto', 'mscmean' or 'mscauto'
    intervals: number of intervals
    no_of_comb: number of interval combinations tested (2, 3 or 4)
    xaxislabels: wavelength/wavenumber labels
    val_method: 'test', 'full', 'syst111', 'syst123', 'random', or 'manual'
    segments: number of samples corresponds to full cv

    Output:
    siModel: structured array containing model information.
    Only RMSECV/RMSEP (siModel.RMSE) as a function of the number of PLS components for each model
    is saved together with an index (siModel.IntComb) of the interval combination.
    siModel.minRMSE stores the minimum RMSECV/RMSEP for each PLS component. The corresponding
    pair of intervals is stored in siModel.minComb

    '''
    import sub_pls_val

    if len(X) == 0 or len(Y) == 0:
        print(' ')
        print(' siModel=sipls(X,Y,no_of_lv,prepro_method,intervals,no_of_comb,xaxislabels,val_method,segments);')
        print(' ')
        print(' Example:')
        print(' siModel=sipls(X,Y,10,''mean'',20,2,[],''syst123'',5);')
        print(' ')
        return

    siModel = {}
    siModel['type'] = 'siPLS'
    siModel['no_of_comb'] = no_of_comb
    siModel['rawX'] = X
    siModel['rawY'] = Y
    siModel['no_of_lv'] = no_of_lv
    siModel['prepro_method'] = prepro_method
    siModel['xaxislabels'] = xaxislabels
    siModel['val_method'] = val_method
    siModel['IntComb'] = []
    siModel['RMSE'] = []
    siModel["minComb"] = []
    siModel["minPLSC"] = []

    if val_method == 'full':
        siModel['segments'] = len(X)
    else:
        siModel['segments'] = segments

    siModel['intervals'] = intervals

    # Calculate intervals
    nint, mint = intervals.shape if isinstance(intervals, np.ndarray) else (1, 1)
    n, m = X.shape
    

    if mint > 1:
        siModel["allint"] = np.column_stack([(np.arange(0, np.round(mint/2)+1, dtype='int')).reshape(-1,1), np.concatenate((intervals[:mint:2], [0])).reshape(-1, 1), np.concatenate((intervals[1:mint:2], [m-1])).reshape(-1, 1)])
        siModel["intervals"] = round(mint/2)
        siModel["intervalsequi"] = 0
    else:
        siModel["intervals"] = intervals
        vars_left_over = m % intervals
        N = np.floor(m/intervals)
        aa = list(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1))
        bb = list(np.arange((vars_left_over-1)*(N+1)+1+1+N-1,m,N))
        if aa == []:
            startint = np.array(bb).reshape(-1,1)
        else:
            startint = np.row_stack([(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1, dtype='int')).reshape(-1,1), np.arange((vars_left_over-1)*(N+1)+1+1+N-1,m,N, dtype='int').reshape(-1,1)])
        endint = np.concatenate((startint[1:intervals, 0]-1, [m-1])).reshape(-1,1)
        siModel["allint"] = np.column_stack((np.arange(0, intervals+1, dtype='int').reshape(-1,1), np.concatenate((startint[:,0], [0])).reshape(-1,1), np.concatenate((endint[:,0], [m-1])).reshape(-1, 1)))
        siModel["intervalsequi"] = 1
    
    # Error checks
    if no_of_comb == 2:
        Total = int(siModel["intervals"]*(siModel["intervals"]-1)/2)
        print(f'In total {Total} models are to be made')
        count = 0
        for i in range(siModel["intervals"]-1):
            for j in range(i+1, siModel["intervals"]+1):
                count += 1
                print(f'Working on model no. {count} of {Total}...')
                int1 = np.arange(int(siModel["allint"][i, 1]), int(siModel["allint"][i, 2])+1).reshape(-1,1)
                int2 = np.arange(int(siModel["allint"][j, 1]), int(siModel["allint"][j, 2])+1).reshape(-1,1)
                selected_vars = np.concatenate((int1, int2))
                #print(f'allint 1 is {int2} allint 2 {siModel["allint"][i, 2]+1}')
                #print(f'sel vars {selected_vars}')
                #print(f'segments {siModel["segments"]}')
                PLSmodel = sub_pls_val.sub_pls_val(siModel["rawX"][:, selected_vars.flatten()], siModel["rawY"],
                                       no_of_lv, prepro_method, val_method, siModel["segments"])
                siModel["IntComb"].append([i, j])
                siModel["RMSE"].append(PLSmodel["RMSE"])
                
    elif no_of_comb == 3:
        Total = int(np.prod(range(siModel["intervals"]-3+1, siModel["intervals"]+1)) / np.prod(range(1, 4)))
        selected_vars = np.zeros((Total, 3), dtype=int)
        print(f'In total {Total} models are to be made')
        count = 0
        for i in range(siModel["intervals"]-1):
            for j in range(i+1, siModel["intervals"]+1):
                for k in range(j+1, siModel["intervals"]+1):
                    count += 1
                    print(f"Working on model no. {count} of {Total}...")
                    int1 = np.arange(siModel["allint"][i, 1], siModel["allint"][i, 2]+1).reshape(-1,1)
                    int2 = np.arange(siModel["allint"][j, 1], siModel["allint"][j, 2]+1).reshape(-1,1)
                    int3 = np.arange(siModel["allint"][k, 1], siModel["allint"][k, 2]+1).reshape(-1,1)
                    selected_vars = np.concatenate((int1, int2, int3))
                    PLSmodel = sub_pls_val.sub_pls_val(siModel["rawX"][:, selected_vars], siModel["rawY"], no_of_lv, prepro_method, val_method, siModel["segments"])
                    siModel["IntComb"].append([i, j, k])
                    siModel["RMSE"].append(PLSmodel['RMSE'])
                    
    elif no_of_comb == 4:  # Four interval models
        Total = int(np.prod(np.arange(siModel["intervals"]-4+1, siModel["intervals"]+1)) / np.prod(np.arange(1, 5)))
        selected_vars = np.zeros((Total, 4))
        print(f'In total {Total} models are to be made')
        count = 0
        for i in range(siModel["intervals"]-1):
            for j in range(i+1, siModel["intervals"]+1):
                for k in range(j+1, siModel["intervals"]+1):
                    for l in range(k+1, siModel["intervals"]+1):
                        count += 1
                        print(f'Working on model no. {count} of {Total}...')
                        int1 = np.arange(siModel["allint"][i, 1], siModel["allint"][i, 2]+1)
                        int2 = np.arange(siModel["allint"][j, 1], siModel["allint"][j, 2]+1)
                        int3 = np.arange(siModel["allint"][k, 1], siModel["allint"][k, 2]+1)
                        int4 = np.arange(siModel["allint"][l, 1], siModel["allint"][l, 2]+1)
                        selected_vars = np.concatenate((int1, int2, int3, int4))
                        PLSmodel = sub_pls_val.sub_pls_val(siModel["rawX"][:, selected_vars], siModel["rawY"], no_of_lv,
                                           prepro_method, val_method, siModel["segments"])
                        siModel["IntComb"].append([i, j, k, l])
                        siModel["RMSE"].append(PLSmodel["RMSE"])
                    
    siModel['IntComb'] = np.array(siModel['IntComb'])
    siModel['RMSE'] = np.array(siModel['RMSE'])
    # for loop equivalent to the above MATLAB code that extracts RMSE values
    RMSE = np.array([siModel["RMSE"][i] for i in range(Total)])

    # First local minima is better; could be changed using e.g. F-test or equal NOT IMPLEMENTED
    # Ones appended to make the search stop if the first local miminum is the last PLSC
    RedRMSE = RMSE[:, 1:]  # PLSC0 is excluded in finding the first local minimum
    SignMat = np.hstack((np.sign(np.diff(RedRMSE, axis=1)), np.ones((Total, 1), dtype=int)))

    min_ix = np.zeros(Total, dtype=int)
    minRMSE = np.zeros(Total)
    for i in range(RedRMSE.shape[0]):
        for j in range(RedRMSE.shape[1]):
            if SignMat[i, j] == 1:
                min_ix[i] = j   # Note: PLSC0 is excluded
                minRMSE[i] = RedRMSE[i, j]  # Note: PLSC0 is excluded
                break

    Index = np.argsort(minRMSE)
    siModel["minRMSE"] = minRMSE[Index]  # Find the lowest RMSEs of total number of models

    for j in range(10):  # Show ten best models
        siModel["minComb"].append(siModel["IntComb"][Index[j]])
        siModel["minPLSC"].append(min_ix[Index[j]])
        
    siModel["minComb"] = np.array(siModel["minComb"])
    siModel["minPLSC"] = np.array(siModel["minPLSC"])
        
    return siModel