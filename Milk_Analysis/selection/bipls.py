import numpy as np
#import sort_ipls
#import sub_iplsreverse 
#from sort_ipls import sort_ipls
#from sub_iplsreverse import sub_iplsreverse, sizer

#from packages import np, sort_ipls, sub_iplsreverse, sizer




def bipls(X,Y,no_of_lv,prepro_method,intervals,xaxislabels,val_method,segments):
    #This function implements the Backward interval based PLS
    import sort_ipls
    import sub_iplsreverse 
    
    # Error checks
    if val_method not in {'test', 'full', 'syst123', 'syst111', 'random', 'manual'}:
        print('Not allowed validation method')
        return None

    if prepro_method not in {'mean', 'auto', 'mscmean', 'mscauto', 'none'}:
        print('Not allowed preprocessing method')
        return None

    if val_method == 'full':
        segments = X.shape[0]

    minRMSEwithout,ix_for_iterative,minRMSEglobal = np.zeros(intervals-1), np.zeros(intervals-1, dtype='int'), np.zeros(intervals-1)
    ModelReverse = sub_iplsreverse.sub_iplsreverse(X,Y,no_of_lv,prepro_method,intervals,xaxislabels,val_method,segments)
    #print(f'Model Reverse a {len(ModelReverse)}')
    #temp1,temp2,minRMSEwithout[0],ix_for_iterative[0],minRMSEglobal[0] = sort_ipls(ModelReverse)
    temp1,temp2,minRMSEwithout[0],ix_for_iterative[0],minRMSEglobal[0] = sort_ipls.sort_ipls(ModelReverse)
    #print(f'min rmse without and global {minRMSEwithout[0]} and {minRMSEglobal[0]} and {ix_for_iterative[0]} and {temp1} and {temp2}')

    newX = np.copy(X)
    RevVars = [newX.shape[1]]
    #print(f'dels is {ModelReverse["allint"][ix_for_iterative[0], 1]}')
    #print(f'ix it {ix_for_iterative[0]}')
    #print(f'dels is {ModelReverse["allint"][ix_for_iterative[0], 1]}')
    dels = np.array(range(ModelReverse['allint'][ix_for_iterative[0], 1],ModelReverse['allint'][ix_for_iterative[0], 2]+1))
    #print(f'dels is {dels}')
    newX = np.delete(newX, dels, axis=1)
    #print(f'newx shape {newX[:,50]}')
    #newX = np.delete(newX, [ModelReverse['allint'][ix_for_iterative, 1]:ModelReverse['allint'][ix_for_iterative, 2]], axis=1)

    #RevIntInfo = np.zeros((intervals-1,1))
    keeptrackofinterval = np.hstack((np.arange(0, intervals).reshape(-1, 1), np.arange(0, intervals).reshape(-1, 1)))
    #print(f'keep track {keeptrackofinterval}')
    RevIntInfo = keeptrackofinterval[ix_for_iterative[0],:]
    #print(f'rev int info {RevIntInfo}')
    #RevIntInfo[0,:] = keeptrackofinterval[ix_for_iterative[0],:]
    keeptrackofinterval = np.delete(keeptrackofinterval, ix_for_iterative[0], axis=0)
    keeptrackofinterval[:, 0] = np.arange(0, intervals-1)
    #print(f'ref int info {RevIntInfo.shape}')
    #print(f'keep track {keeptrackofinterval}')
    
    for i in range(1, intervals-1):
        RevVars.append(newX.shape[1])
        #print(f'intervals i is {intervals-(i-1+1)}')
        #print(f'y is {Y}')
        #print(f'segment is {segments}')
        #print(f'no of lv is {no_of_lv}')
        ModelReverse = sub_iplsreverse.sub_iplsreverse(newX, Y, no_of_lv, prepro_method, intervals-(i-1+1), xaxislabels, val_method, segments)
        #print(f'model reverse is {ModelReverse["PLSmodel"][1]}')
        temp1,temp2,minRMSEwithout[i],ix_for_iterative[i],minRMSEglobal[i] = sort_ipls.sort_ipls(ModelReverse)
        newX = np.delete(newX, np.array(range(ModelReverse['allint'][ix_for_iterative[i], 1],ModelReverse['allint'][ix_for_iterative[i], 2]+1)), axis=1)
        #RevIntInfo[i,:] = keeptrackofinterval[ix_for_iterative[i],:]
        #print(f'{i} and rev i is {RevIntInfo} and {keeptrackofinterval} and ix int {ix_for_iterative[i]} and rmse {temp1}')
        #print(f'ref int info {RevIntInfo}')
        #print(f'keep track {keeptrackofinterval}')
        RevIntInfo = np.vstack((RevIntInfo, keeptrackofinterval[ix_for_iterative[i],:]))
        keeptrackofinterval = np.delete(keeptrackofinterval, ix_for_iterative[i], axis=0)
        keeptrackofinterval[:, 0] = np.arange(0, intervals-i-1)
    
    l = len(minRMSEwithout)
    #print(f'min RMSE {minRMSEwithout}')
    #print(f'min RMSE global {minRMSEglobal}')
    #RevRMSE = np.concatenate((minRMSEglobal.reshape(-1, 1), minRMSEwithout[l-1].reshape(-1, 1)))
    RevRMSE = np.concatenate((minRMSEglobal.T, np.array([minRMSEwithout[l-1]])))
    #print(f'revrmse {RevRMSE}')
    #print(f'ref int info {RevIntInfo}')
    RevIntInfo = np.delete(RevIntInfo, 0, axis=1)
    #print(f'ref int info {RevIntInfo}')
    RevIntInfo = np.vstack((RevIntInfo, keeptrackofinterval[0,1]))
    #RevIntInfo[intervals-1] = keeptrackofinterval[0, 1]
    #if intervals==2:
    #    RevIntInfo = np.hstack((RevIntInfo, keeptrackofinterval[0, 1]))
    #else:
    #    RevIntInfo[intervals-1] = keeptrackofinterval[0, 1]
    #print(f'ref int info after {RevIntInfo}')
    RevVars.extend([newX.shape[1]])
    #print(f'ref vars {RevVars}')

    n,m = X.shape
    nint,mint = intervals.shape if isinstance(intervals, np.ndarray) else (1, 1)
    if mint > 1:
        #allint = np.column_stack([(np.arange(1, np.round(mint/2)+1))', np.concatenate((intervals[:mint:2], [1])), np.concatenate((intervals[1::2], [m]))])
        allint = np.column_stack([(np.arange(0, np.round(mint/2)+1)).reshape(-1,1), np.concatenate((intervals[:mint:2], [0])).reshape(-1, 1), np.concatenate((intervals[1:mint:2], [m-1])).reshape(-1, 1)])
        intervals = np.round(mint/2)
        intervalsequi = 0
    else:
        vars_left_over = m % intervals
        N = m // intervals
        # Distributes vars_left_over in the first "vars_left_over" intervals
        #startint = np.column_stack([(np.arange(0, (N+1)*(vars_left_over-1)+2, N+1)).reshape(-1, 1), np.arange(((vars_left_over-1)*(N+1)+1), ((vars_left_over-1)*(N+1)+1)+N+1)])
        #print(f'start int a {(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1))}')
        #print(f'start int b {np.arange((vars_left_over-1)*(N+1)+1+1+N-1,m,N)}')
        aa = list(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1))
        bb = list(np.arange((vars_left_over-1)*(N+1)+1+1+N-1,m,N))
        if aa == []:
            startint = np.array(bb).reshape(-1,1)
        else:
            startint = np.row_stack([(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1)).reshape(-1,1), np.arange((vars_left_over-1)*(N+1)+1+1+N-1,m,N).reshape(-1,1)])
            #np.column_stack([(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1)).reshape(-1,1), np.arange((vars_left_over-1)*(N+1)+1+1+N-1,m,N).reshape(-1,1)])
        endint = np.concatenate((startint[1:intervals, 0]-1, [m-1])).reshape(-1,1)
        #print(f'end int {endint}')
        #print(f'start int {startint}')
        #print(f'all int a {(np.arange(0, intervals+1)).reshape(-1, 1)}')
        #print(f'all int b {np.concatenate((startint[:,0], [1])).reshape(-1,1)}')
        #print(f'all int c {np.concatenate((endint[:,0], [m-1])).reshape(-1, 1)}')
        allint = np.column_stack([(np.arange(0, intervals+1)).reshape(-1, 1), np.concatenate((startint[:,0], [0])).reshape(-1,1), np.concatenate((endint[:,0], [m-1])).reshape(-1, 1)])
        #print(f'all int {allint}')
        intervalsequi = 1

    biModel = {'type': 'biPLS', 'rawX': X, 'rawY': Y, 'no_of_lv': no_of_lv, 'prepro_method': prepro_method, 
           'intervals': intervals, 'allint': allint, 'intervalsequi': intervalsequi, 'xaxislabels': xaxislabels,
           'val_method': val_method, 'segments': segments, 'RevIntInfo': RevIntInfo.flatten(), 'RevRMSE': RevRMSE,
           'RevVars': RevVars}
    #print(f'bimodel is {biModel}')

    return biModel

