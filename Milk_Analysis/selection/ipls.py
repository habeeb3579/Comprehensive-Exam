#from sub_pls_val import sub_pls_val
#import numpy as np
#from sub_iplsreverse import sizer

#import sub_pls_val
#import numpy as np
#import sub_iplsreverse 
#from packages import np, sub_pls_val, sizer
def ipls(X, Y, no_of_lv, prepro_method, intervals, xaxislabels, val_method, segments):
    
    ''' ipls calculates the interval models based on PLS
    #
    # Input:
    # X is the independent variables
    # Y is the dependent variable(s), NOTE: Y is allways autoscaled
    # no_of_lv is the maximum number of PCA or PLS components
    # prepro_method (for X only) is 'mean', 'auto', 'mscmean' or 'mscauto'
    #    Note: msc is performed in each interval
    # intervals is the number of intervals
    #    if intervals is a row vector divisions are made based on the elements
    #    [startint1 endint1 startint2 endint2 startint3 endint3], see an example in manint
    # xaxislabels (self explainable), if not available type []
    # val_method is 'test', 'full', 'syst111', 'syst123', 'random', or
    #    'manual'; the last five are cross validation based methods
    # segments (segments = number of samples corresponds to full cv)
    #    if intervals is a cell array cross validation is performed according
    #    to this array, see the script makeManualSegments
    #
    # Output:
    # Model is a structured array containing all model information
    #
    #
    # Model=ipls(X,Y,no_of_lv,prepro_method,intervals,xaxislabels,val_method,segments);

    # Functions used: sub_pls_val '''
    
    import sub_pls_val
    import numpy as np
    import sub_iplsreverse 

    if X is None or Y is None:
        print("Model=ipls(X,Y,no_of_lv,prepro_method,intervals,xaxislabels,val_method,segments);")
        print("Example:")
        print("Model=ipls(X,Y,7,'mean',20,xaxis,'syst123',5);")
        return

    # Error checks
    if val_method not in ['test', 'full', 'syst123', 'syst111', 'random', 'manual']:
        print("Not allowed validation method")
        return None

    if prepro_method not in ['mean', 'auto', 'mscmean', 'mscauto', 'none']:
        print("Not allowed preprocessing method")
        return None

    if val_method.lower() == 'manual' and not isinstance(segments, list):
        print("You need to specify the manual segments in a list, see makeManualSegments")
        return None

    if val_method.lower() == 'manual' and isinstance(segments, list):
        Nsamples = sum([len(seg) for seg in segments])
        if X.shape[0] != Nsamples:
            print("The number of samples in X does not correspond to the total number of samples in segments")
            return None
    # End error checks

    Model = {}
    Model['type'] = 'iPLS'
    Model['rawX'] = X
    Model['rawY'] = Y
    Model['no_of_lv'] = no_of_lv
    Model['prepro_method'] = prepro_method
    Model['xaxislabels'] = xaxislabels  # Final use of xaxislabels in this file; i.e. if reversed axes are present this should be taken care of in iplsplot
    Model['val_method'] = val_method
    Model["PLSmodel"] = []
    
    n, m = X.shape
    if Model["val_method"].lower() == 'full' or segments==None:
        Model["segments"] = n
    else:
        Model["segments"] = segments

    nint, mint = sub_iplsreverse.sizer(intervals)
    if mint > 1:
        Model["allint"] = np.vstack([(np.arange(0, mint//2+1, dtype='int')).T,
        np.vstack((intervals[::2].T, np.array([0]))),
        np.vstack((intervals[1::2].T, np.array([m-1])))])
        Model["intervals"] = mint//2
        Model["intervalsequi"] = 0
    else:
        Model["intervals"] = intervals
        vars_left_over = m % intervals
        N = m // intervals
        aa = list(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1, dtype='int'))
        bb = list(np.arange((vars_left_over-1)*(N+1)+1+1+N-1,m,N, dtype='int'))
        if aa == []:
            startint = np.array(bb).reshape(-1,1)
        else:
            startint = np.row_stack([(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1, dtype='int')).reshape(-1,1),
            np.arange((vars_left_over-1)*(N+1)+1+N, m, N, dtype='int').reshape(-1,1)])
        endint = np.concatenate((startint[1:intervals, 0]-1, [m-1])).reshape(-1,1)
        #print(f'start int {startint}')
        #print(f'end int {endint}')
        #print(f'all int a {np.column_stack((np.arange(0, intervals+1).reshape(-1,1), np.concatenate((startint[:,0],[0])).reshape(-1,1)))}')
        #print(f'ender {np.concatenate((endint[:,0], [m-1])).reshape(-1,1)}')
        Model["allint"] = np.column_stack([np.arange(0, intervals+1, dtype='int').reshape(-1,1), np.concatenate((startint[:,0],[0])).reshape(-1,1), np.concatenate((endint[:,0], [m-1])).reshape(-1,1)])
        Model["intervalsequi"] = 1

    for i in range(Model["allint"].shape[0]):
        if i < Model["allint"].shape[0]-1:
            print(f"Working on interval no. {i} of {Model['allint'].shape[0]-2}...")
        else:
            print("Working on full spectrum model...")
        dels2 = np.array(range(Model['allint'][i, 1],Model['allint'][i, 2]+1))
        Model["PLSmodel"].append(sub_pls_val.sub_pls_val(X[:, dels2], Y, Model["no_of_lv"], Model["prepro_method"],
        Model["val_method"], Model["segments"]))
        
    return Model