import numpy as np
def sub_bipls_vector(biplslimitModel, OrigVars, OrigIntervals):
    # bipls_vector
    # Input:
    # biplslimitModel
    # OrigVars: number of original variables (e.g. 926)
    # OrigIntervals: number of original intervals (e.g. 20)
    # Output:
    # ix_vector: index of the variables selected
    #
    # ix_vector=sub_bipls_vector(biplslimitModel,OrigVars,OrigIntervals);
    #
    # Functions used: none
    
    if biplslimitModel==None:
        print(" ")
        print(" ix_vector=sub_bipls_vector(biplslimitModel,OrigVars,OrigIntervals);")
        print(" ")
        print(" ix_vector=sub_bipls_vector(biplslimitModel,926,20));")
        print(" ")
        return
    
    ix = np.where(biplslimitModel["RevVars"] < 400)[0] #[i for i, var in enumerate(biplslimitModel["RevVars"]) if var < 400]
    ix = biplslimitModel["RevIntInfo"][ix]
    
    nint, mint = OrigIntervals.shape if isinstance(OrigIntervals, np.ndarray) else (1, 1)
    if mint > 1:
        allint = np.column_stack([(np.arange(0, np.round(mint/2)+1)).reshape(-1,1), np.concatenate((OrigIntervals[:mint:2], [0])).reshape(-1, 1), np.concatenate((OrigIntervals[1:mint:2], [OrigVars-1])).reshape(-1, 1)])
        #allint = [[i, j, k] for i, j, k in zip(range(1, round(mint/2) + 2), 
        #                                        OrigIntervals[::2, 0], 
        #                                        [1] + OrigIntervals[1::2, 0].tolist() + [OrigVars])]
        OrigIntervals = round(mint/2)
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
        
        
        #vars_left_over = OrigVars % OrigIntervals
        #N = OrigVars // OrigIntervals
        # Distributes vars_left_over in the first "vars_left_over" intervals
        #startint = [[i] for i in range(1, vars_left_over * (N + 1) + 1, N + 1)]
        #startint += [[i] for i in range(vars_left_over * (N + 1) + 1 + N, OrigVars, N)]
        #endint = startint[1:] + [[OrigVars]]
        #allint = [[i, j[0], k[0]] for i, j, k in zip(range(1, OrigIntervals + 2), startint, endint)]
        
    ix_vector = []
    for i in range(len(ix)):
        ix_vector.append(list(range(allint[ix[i], 1], allint[ix[i],2] + 1)))
        #ix_vector += list(range(allint[ix[i] - 1][1], allint[ix[i] - 1][2] + 1))
    ix_vector = np.array(ix_vector)
    ix_vector = np.sort(ix_vector)
    
    return ix_vector