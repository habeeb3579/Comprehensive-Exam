import numpy as np
from preprop import *
#from sub_pls_pre import sub_pls_pre
#import sub_pls_pre

#from packages import sub_pls_pre

def plspredict(Xpred, Model, no_of_lv, Yref=None):
    """
    plspredict predicts reference values for new X data
    Args:
    Xpred: test data 
    Model: output from plsmodel.m
    no_of_lv: number of PLS components to use in prediction
    Yref: optional, reference values (if available)

    Returns:
    predModel: a structure array containing all model information
    """
    import sub_pls_pre
    # Error checks
    if Model['type'] != 'PLS':
        print('This function only works with output from plsmodel.m')
        predModel = {}
        return predModel

    predModel = {'type': 'PLSprediction'}

    if 'windowsize' in Model:
        # If oneModel is based on mwModel
        selected_vars = Model['selected_vars']
    else:
        selected_vars = []
        for i in range(len(Model['selected_intervals'])):
            temp = np.arange(Model['allint'][Model['selected_intervals'][i]-1, 1], 
                         Model['allint'][Model['selected_intervals'][i]-1, 2]+1)
            #selected_vars = np.concatenate([selected_vars, temp])
            selected_vars.extend(list(temp)) 
        
    predModel['Xpred'] = Xpred[:, selected_vars] # From input
    predModel['no_of_lv'] = no_of_lv # From input

    if Yref is not None:
        predModel['Yref'] = Yref # From input
    else:
        predModel['Yref'] = []

    # Transformations - X
    if Model['prepro_method'].lower() == 'mean':
        Xtrans_cal, mx = mncn(Model['rawX'][:, selected_vars])
        Xpred = scalenew(predModel['Xpred'], mx)
    elif Model['prepro_method'].lower() == 'auto':
        Xtrans_cal, mx, stdx = auto(Model['rawX'][:, selected_vars])
        Xpred = scalenew(predModel['Xpred'], mx, stdx)
    elif Model['prepro_method'].lower() == 'mscmean':
        Xtrans_cal, Xmean_cal = msc(Model['rawX'][:, selected_vars])
        Xtrans_cal, mx = mncn(Xtrans_cal)
        Xpred = msc_pre(predModel['Xpred'], Xmean_cal)
        Xpred = scalenew(predModel['Xpred'], mx)
    elif Model['prepro_method'].lower() == 'mscauto':
        Xtrans_cal, Xmean_cal = msc(Model['rawX'][:, selected_vars])
        Xtrans_cal, mx, stdx = auto(Xtrans_cal)
        Xpred = msc_pre(predModel['Xpred'], Xmean_cal)
        Xpred = scalenew(predModel['Xpred'], mx, stdx)
    elif Model['prepro_method'].lower() == 'none':
        # To secure that no centering/scaling is OK
        Xpred = predModel['Xpred']

    Ypred = sub_pls_pre.sub_pls_pre(Xpred, Model['PLSmodel'][0]['bsco'], Model['PLSmodel'][0]['P'], 
                    Model['PLSmodel'][0]['Q'], Model['PLSmodel'][0]['W'], no_of_lv)

    predModel['Ypred0'] = np.ones((Xpred.shape[0], 1)) * np.mean(Model['rawY']) # For zero PLSC estimate as average of calibration segment

    # Back transformation - Y
    Ytrans_cal, my, stdy = auto(Model["rawY"])
    for i in range(no_of_lv):
        predModel["Ypred[:, :, i]"] = scaleback(Ypred[:, :, i], my, stdy)

    if Yref is not None:
        RMSE = np.zeros(no_of_lv)
        Bias = np.zeros(no_of_lv)
        for i in range(no_of_lv):
            RMSE[i], Bias[i] = rmbi(Yref, predModel.Ypred[:, :, i])

        RMSE0, Bias0 = rmbi(Yref, predModel.Ypred0)
        predModel["RMSE"] = np.concatenate(([RMSE0], RMSE))
        predModel["Bias"] = np.concatenate(([Bias0], Bias))
    else:
        predModel["RMSE"] = []
        predModel["Bias"] = []
    predModel["CalModel"] = Model #From input
    return predModel