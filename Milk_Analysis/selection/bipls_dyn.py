import numpy as np
import matplotlib.pyplot as plt
from time import time
#import bipls_table
#import sub_bipls_limit
#import sub_bipls_vector_limit
#from bipls_table import biplstable
#from sub_bipls_limit import sub_bipls_limit
#from sub_bipls_vector_limit import sub_bipls_vector_limit
#from packages import np, plt, time, biplstable, sub_bipls_limit, sub_bipls_vector_limit
def sort_with_index(arr):
    """
    Sorts a NumPy array and returns the sorted array and index.
    """
    # Get the sorted index
    idx = np.argsort(arr)
    # Use the index to sort the array
    sorted_arr = arr[idx]
    return sorted_arr, idx


def bipls_dyn(dataset, nooflv, seqint, MinNoVars, varold=None, c=None):
    '''
    bipls_dyn makes dynamic biPLS calculations with a number of intervals defined by the user.
    The output 'var' is plotted at the end of the calculation.
    NOTE: Calculations performed with 'syst123' and 'auto'
    
    Input:
        dataset: X and y in the same matrix; dataset=[X y];
        nooflv: number of latent variables to calculate in each model
        seqint: sequence of intervals in the runs, e.g. [16:25] or 20
        MinNoVars is the number of variables to stop at in dynamic biPLS, e.g. 400
        varold (optional) is the vector var obtained in previous runs, to which the results of the new runs will be added
        c (optional) is the number of previous runs 
    Output:
        var: vector containing the number of times each variables has been retained
    
    After each cycle the variables var and c (the number of performed cycles) are saved.
    This allows not to waste the partial results in case of a system failure and to add new runs.
    
    '''
    import bipls_table
    import sub_bipls_limit
    import sub_bipls_vector_limit
    
    # Function used: sub_bipls_limit, sub_bipls_vector_limit (both in subfunction iplsprega)
    
    if dataset==None:
        print('\nvar=bipls_dyn(dataset,nooflv,seqint,MinNoVars,varold,c);\n')
        print('Example:')
        print('var=bipls_dyn(dataset,15,[16:25],400,varold,10);\n')
        print('This means that a maximum of 15 latent variables are computed, the runs will be made on 16, 17, ..., 25 intervals')
        print('and no more than 400 variables will be retained; 10 runs have already been performed, and the number of times each')
        print('variable has been retained is stored in the vector varold\n')
        return
    
    randomiz()  # Subfunction of this file
    
    o, v = dataset.shape
    v = v - 1
    
    if (varold==None and c==None):
        c = 0
        var = np.zeros(v)
    else:
        var = varold
    
    for i in range(len(seqint)):
        c += 1
        k = np.random.permutation(o)
        vv = iplsprega(dataset[k, :], nooflv, seqint[i], MinNoVars)
        for j in range(len(vv)):
            var[vv[j]] += 1
        plt.close()
        plt.plot(var)
        plt.xlim([1, v])
        plt.ylim([0, max(var) + 0.1])
        #a, b = zip(*sorted(zip(var, range(len(var)))))
        a, b = sort_with_index(var)
        a = a[::-1]
        plt.plot([1, v], [a[MinNoVars] + 0.5, a[MinNoVars] + 0.5], '--')
        plt.xlabel('Wavelength number')
        plt.title('Frequency of selections after ' + str(c) + ' cycles')
        plt.show(block=False)
        plt.draw()
        plt.pause(0.001)
        np.save('var.npy', var)
        np.save('c.npy', c)



def iplsprega(dataset, nooflv, numint, MinNoVars):
    """
    iplsprega performs iPLS in its backward form as a wavelength elimination tool.
    This is done preliminary to the application of GA (gaplssp).

    Input:
        dataset: in the form usual for GA, with the y variable as the last variable
        nooflv: number of latent variables to calculate in each model
        numint: the number of intervals
        MinNoVars is the number of variables to stop at in biPLS
    Output:
        retvar: a vector with the indexes of the retained variables
    """
    if dataset==None:
        print(' ')
        print(' retvar=iplsprega(dataset,nooflv,numint,MinNoVars);')
        print(' ')
        print(' Example:')
        print(' retvar=iplsprega(dataset,12,20,400);')
        print(' ')
        return

    x = dataset[:, :-1]
    y = dataset[:, -1]
    biplslimitModel = sub_bipls_limit.sub_bipls_limit(x, y, nooflv, 'auto', numint, MinNoVars, [], 'syst123', 5)
    # max. nooflv comp., numint int., autosc., 5 del.gr. with venetian blinds.

    bipls_table.biplstable(biplslimitModel)  # prints the table of the results

    # the vector retvar stores the variables retained by backward iPLS
    retvar = sub_bipls_vector_limit.sub_bipls_vector_limit(biplslimitModel, x.shape[1], numint)
    return retvar


def randomiz():
    np.random.seed(sum(100 * np.array(time.localtime())))