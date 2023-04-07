#import numpy as np
#from sub_pls_val import sub_pls_val
#from sub_pls_val import sub_pls_val

import numpy as np
#import sub_pls_val
#import sub_pls_val

#from packages import sub_pls_val, np

def sizer(seg):
    return seg.shape if isinstance(seg, (list, tuple, np.ndarray)) else (1, 1)

def sub_iplsreverse(X, Y, no_of_lv, prepro_method, intervals, xaxislabels, val_method, segments):
    """
    sub_iPLSreverse calculates iPLS models WITHOUT the given interval (opposite of ipls)

    Input:
    X is the independent variables
    Y is the dependent variable(s), NOTE: Y is always autoscaled
    no_of_lv is the maximum number of PLS components
    intervals is the number of intervals
       if intervals is a row vector divisions are made based on the elements
       [startint1 endint1 startint2 endint2 startint3 endint3], see an example in manint
    val_method is 'test' 'full' 'syst111', 'syst123', 'random' or 'manual'
    prepro_method (for X only) is 'mean', 'auto', 'mscmean' or 'mscauto'
    xaxislabels (self explainable), if not available type []
    segments (segments = number of samples corresponds to full cv)

    Output:
    ModelsReverse is a structured array containing all model information
    """
    import sub_pls_val
    import sub_pls_val
    ModelsReverse = {'rawX': X, 'rawY': Y, 'no_of_lv': no_of_lv, 'xaxislabels': xaxislabels,
                     'val_method': val_method, 'segments': segments, 'prepro_method': prepro_method,
                     'PLSmodel':[]}

    n, m = X.shape
    if val_method == 'full':
        ModelsReverse['segments'] = n

    # Error checks
    #if isinstance(segments, (list, tuple, np.ndarray)) and val_method == 'manual':
    if isinstance(segments, (list, tuple, np.ndarray)) and val_method == 'manual':
        Nsamples = sum(np.max(sizer(x)) for x in segments)
        if X.shape[0] != Nsamples:
            print('The number of samples in X does not correspond to the number of samples in segments')
            return

    nint, mint = sizer(intervals)
    #print(f'mint is {mint}')
    if mint > 1:
        ModelsReverse['allint'] = np.hstack([(np.arange(round(mint / 2) + 1)).reshape(-1, 1), 
                                             np.concatenate((intervals[:mint:2], [0])).reshape(-1, 1),
                                             np.concatenate((intervals[1:mint:2], [m-1])).reshape(-1, 1)])
                                              #np.vstack([intervals[:mint:2, 0], np.ones(round(mint / 2)).astype(int)]).T,
                                              #np.vstack([intervals[1::2, 0], np.array([m])]).T])
        ModelsReverse['intervals'] = round(mint / 2)
        ModelsReverse['intervalsequi'] = 0
    else:
        ModelsReverse['intervals'] = intervals
        vars_left_over = m % intervals
        N = m // intervals
        # Distributes vars_left_over in the first "vars_left_over" intervals
        #print(f'n is {N}')
        #print(f'm is {m}')
        #print(f'intervals is {intervals}')
        #print(f'vars_left_over is {vars_left_over}')
        #print(f'a start int {(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1))}')
        #print(f'a start int b {np.arange((vars_left_over-1)*(N+1)+1+1+N-1,m,N)}')
        startint = np.row_stack([(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1)).reshape(-1,1), np.arange((vars_left_over-1)*(N+1)+1+1+N-1,m,N).reshape(-1,1)])
        endint = np.concatenate((startint[1:intervals, 0]-1, [m-1]))
        #print(f'start int real {startint}')
        #print(f'end int real {endint}')
        ModelsReverse['allint'] = np.column_stack([(np.arange(0, intervals+1)).reshape(-1, 1), np.concatenate((startint[:, 0], [0])).reshape(-1,1), np.concatenate((endint, [m-1])).reshape(-1, 1)])
        #print(f'end int real {ModelsReverse["allint"]}')
        ModelsReverse['intervalsequi'] = 1

    #print(f'model reverse {ModelsReverse}')
    # Local calibration
    GlobalIntervalInformation = list(range(ModelsReverse['allint'][ModelsReverse['intervals'],1], ModelsReverse['allint'][ModelsReverse['intervals'],2]+1))
    #print(f'Global interval {GlobalIntervalInformation}')
    #ModelsReverse.allint[ModelsReverse.intervals+1:, 2:4]
    #print(f'model all int {ModelsReverse["allint"]}')
    for i in range(ModelsReverse['allint'].shape[0]):
        if i < ModelsReverse['allint'].shape[0]-1:
            print(f'Working on interval no. {i} of {ModelsReverse["allint"].shape[0]-2}...')
        else:
            print('Working on full spectrum calibration...')
        GlobalWithoutIntervalWithout1 = GlobalIntervalInformation.copy()
        del GlobalWithoutIntervalWithout1[ModelsReverse['allint'][i, 1]:ModelsReverse['allint'][i, 2]+1]
        GlobalWithoutIntervalWithout = np.array(GlobalWithoutIntervalWithout1.copy())
        #if i==0 or i==1 or i==2:
            #print(f'global without {i} is {GlobalWithoutIntervalWithout}')
        if i == ModelsReverse['intervals']:
            dels2 = np.array(range(ModelsReverse['allint'][i, 1],ModelsReverse['allint'][i, 2]+1))
            #ModelsReverse['PLSmodel'][i] = sub_pls_val(X[:, ModelsReverse['allint'][i, 1]:ModelsReverse['allint'][i, 2]+1], Y, no_of_lv, prepro_method, val_method, segments)
            #ModelsReverse['PLSmodel'].append(sub_pls_val(X[:, ModelsReverse['allint'][i, 1]:ModelsReverse['allint'][i, 2]+1], Y, no_of_lv, prepro_method, val_method, segments))
            ModelsReverse['PLSmodel'].append(sub_pls_val.sub_pls_val(X[:, dels2], Y, no_of_lv, prepro_method, val_method, segments))
        else:
            #ModelsReverse['PLSmodel'][i] = sub_pls_val(X[:, GlobalWithoutIntervalWithout], Y, no_of_lv, prepro_method, val_method, segments)
            ModelsReverse['PLSmodel'].append(sub_pls_val.sub_pls_val(X[:, GlobalWithoutIntervalWithout], Y, no_of_lv, prepro_method, val_method, segments))

    return ModelsReverse 