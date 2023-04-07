#import numpy as np
#from sub_pls_val import sub_pls_val
#from sub_iplsreverse import sizer 

import numpy as np
#import sub_pls_val
#import sub_iplsreverse 


#from packages import np, sub_pls_val, sizer
def plsmodel(Model, selected_intervals, no_of_lv, prepro_method, val_method, segments=None):
    '''  plsmodel calculates a combined PLS model from selected interval(s)
    #
    #  Input:
    #  Model: output from ipls, bipls, mwpls or sipls
    #  selected_intervals are the interval numbers arranged in a vector
    #  no_of_lv is the maximum number of PLS components
    #  prepro_method: 'mean', 'auto', 'mscmean' or 'mscauto'
    #  val_method is 'test', 'full', 'syst111', 'syst123', 'random', or 'manual'
    #  segments (segments = number of samples corresponds to full cv)
    #
    #  Output:
    #  oneModel is a structured array containing all model information
    #
    #
    #  oneModel=plsmodel(Model,selected_intervals,no_of_lv,prepro_method,val_method,segments);
    '''
    import sub_pls_val
    import sub_iplsreverse
    
    if not Model:
        print(' ')
        print(' oneModel=plsmodel(Model,selected_intervals,no_of_lv,prepro_method,val_method,segments);')
        print(' ')
        print(' Example:')
        print(' oneModel=plsmodel(Model,[9 11 14 18],10,''mean'',''syst123'',5);')
        print(' ')
        return

    oneModel = {'type': 'PLS'}
    oneModel['rawX'] = Model['rawX']  # From Model
    oneModel['rawY'] = Model['rawY']  # From Model
    oneModel['no_of_lv'] = no_of_lv  # From input
    oneModel['prepro_method'] = prepro_method  # From input
    oneModel['xaxislabels'] = Model['xaxislabels']  # From Model
    oneModel['val_method'] = val_method  # From input

    if val_method.lower() == 'full' and segments== None:
        oneModel['segments'] = Model['rawX'].shape[0]
    elif val_method.lower() == 'full' and segments != None:
        oneModel['segments'] = Model['rawX'].shape[0]
    else:
        oneModel['segments'] = segments

    if Model['type'] in {'iPLS', 'biPLS', 'siPLS'}:
        oneModel['intervals'] = 1  # As default
        oneModel['selected_intervals'] = selected_intervals  # From input
        oneModel['allint'] = Model['allint']  # From Model
    elif Model['type'] == 'Moving window':
        oneModel['intervals'] = 1  # As default
        oneModel['selected_intervals'] = selected_intervals  # From input
        oneModel['selected_vars'] = []
        oneModel['windowsize'] = Model['windowsize']

    # Error checks
    if val_method.lower() == 'manual' and np.max(sub_iplsreverse.sizer(oneModel['segments'])) == 1:
        print('You need to specify manual segments')
        return

    # Only for manual cross validation
    if np.max(sub_iplsreverse.sizer(oneModel['segments'])) > 1 and val_method.lower() == 'manual':
        Nsamples = sum([np.max(sub_iplsreverse.sizer(x)) for x in oneModel['segments']])
        if Model['rawX'].shape[0] != Nsamples:
            print('The number of samples in X does not correspond to the number of samples in manualseg')
            return

    if Model["type"] == 'Moving window':
        if (selected_intervals < 1) or (selected_intervals > Model.rawX.shape[1]):
            print('Not allowed interval number')
            return
        if np.max(selected_intervals.shape) > 1:
            print('Only one interval can be selected for moving window models')
            return

    #print(f'model all int {Model["allint"]}')
    if Model["type"] in ['iPLS', 'biPLS', 'siPLS']:
        selected_vars = []  
        for i in range(len(selected_intervals)):
            temp = np.arange(Model["allint"][selected_intervals[i]-1, 1], Model["allint"][selected_intervals[i]-1, 2]+1, dtype=int)
            selected_vars.extend(list(temp)) 
            #= np.concatenate((selected_vars, temp))
    elif Model["type"] == 'Moving window':
        selected_vars = np.arange(selected_intervals - np.floor(Model.windowsize/2)-1, selected_intervals + np.floor(Model.windowsize/2) + 1, dtype=int)
        if selected_vars[0] < 0:
            selected_vars = np.arange(0, selected_intervals + np.floor(Model.windowsize/2) + 1, dtype=int)
        elif selected_vars[-1] > Model.rawX.shape[1]:
            selected_vars = np.arange((selected_intervals - np.floor(Model.windowsize/2))-1, Model.rawX.shape[1], dtype=int)
        oneModel['selected_vars'] = selected_vars
    print(f'sel var {selected_vars}')
    #print(f'x is {Model["rawX"][:,selected_vars]}')
    #print(f'y is {Model["rawY"]}')
    #print(f'segment is {oneModel["segments"]}')
    oneModel['PLSmodel'] = [sub_pls_val.sub_pls_val(Model["rawX"][:,selected_vars], Model["rawY"], no_of_lv, prepro_method, val_method, oneModel["segments"])]
    return oneModel