def sub_bipls_vector_limit(biplslimitModel, OrigVars, OrigIntervals):

    # sub_bipls_vector_limit
    # Input:
    #   biplslimitModel (outout from bipls_limit.m)
    #   OrigVars: number of original variables (e.g. 926)
    #   OrigIntervals: number of original intervals (e.g. 20)
    # Output:
    #   ix_vector: index of the variables selected
    #
    # ix_vector=sub_bipls_Vector_Limit(biplslimitModel,OrigVars,OrigIntervals));

    # Functions used: none

    if not biplslimitModel:
        print(' ')
        print('ix_vector=sub_bipls_vector_limit(biplslimitModel,OrigVars,OrigIntervals));')
        print(' ')
        print('ix_vector=sub_bipls_vector_limit(biplslimitModel,926,20));')
        print(' ')
        return

    # int_ix=size(biplslimitModel.RevIntInfo,1); % New
    int_ix = None
    for i in range(len(biplslimitModel["RevIntInfo"])):
        if biplslimitModel["RevIntInfo"][i] == 0:
            int_ix = i - 1
            break

            
    nint,mint = OrigIntervals.shape if isinstance(OrigIntervals, np.ndarray) else (1, 1)
    if mint > 1:
        #allint = np.column_stack([(np.arange(1, np.round(mint/2)+1))', np.concatenate((intervals[:mint:2], [1])), np.concatenate((intervals[1::2], [m]))])
        allint = np.column_stack([(np.arange(0, np.round(mint/2)+1)).reshape(-1,1), np.concatenate((OrigIntervals[:mint:2], [0])).reshape(-1, 1), np.concatenate((OrigIntervals[1:mint:2], [OrigVars-1])).reshape(-1, 1)])
        OrigIntervals = np.round(mint/2)
    else:
        vars_left_over = OrigVars % OrigIntervals
        N = OrigVars // OrigIntervals
        aa = list(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1))
        bb = list(np.arange((vars_left_over-1)*(N+1)+1+1+N-1,OrigVars,N))
        if aa == []:
            startint = np.array(bb).reshape(-1,1)
        else:
            startint = np.row_stack([(np.arange(0, (vars_left_over-1)*(N+1)+1, N+1)).reshape(-1,1), np.arange((vars_left_over-1)*(N+1)+1+1+N-1,OrigVars,N).reshape(-1,1)])
        endint = np.concatenate((startint[1:OrigIntervals, 0]-1, [OrigVars-1])).reshape(-1,1)
        allint = np.column_stack([(np.arange(0, OrigIntervals+1)).reshape(-1, 1), np.concatenate((startint[:,0], [0])).reshape(-1,1), np.concatenate((endint[:,0], [OrigVars-1])).reshape(-1, 1)])
    
    
    #nint, mint = OrigIntervals.shape
    #if mint > 1:
    #    allint = np.vstack([(np.arange(round(mint/2)+1)+1), np.concatenate(([OrigIntervals[::2, 0]], [1])), np.concatenate(([OrigIntervals[1::2, 0]], [OrigVars]))]).T
    #    OrigIntervals = round(mint/2)
    #else:
    #    vars_left_over = OrigVars % OrigIntervals
    #    N = OrigVars // OrigIntervals
        # Distributes vars_left_over in the first "vars_left_over" intervals
    #    startint = np.vstack((np.arange(1, (vars_left_over-1)*(N+1)+2, N+1), np.arange((vars_left_over-1)*(N+1)+1+1+N, OrigVars, N))).T
    #    endint = np.concatenate((startint[1:OrigIntervals, 1]-1, [OrigVars]))
    #    allint = np.vstack((np.arange(OrigIntervals+1)+1, np.concatenate(([startint[:, 0]], [1])), np.concatenate(([endint], [OrigVars])))).T

    out_vector = []
    for i in range(int_ix):
        out_vector.extend(list(range(allint[biplslimitModel["RevIntInfo"][i], 1], allint[biplslimitModel["RevIntInfo"][i], 2]+1)))
    out_vector = np.array(out_vector)
    ix_vector = np.arange(OrigVars)
    ix_vector = np.delete(ix_vector, out_vector)
    ix_vector = np.sort(ix_vector)

    return ix_vector