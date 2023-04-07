import matplotlib.pyplot as plt
import numpy as np
from preprop import rmbi

def plspvsm(Model,no_of_int_lv,interval=None,y_variable=None):
    # plspvsm plots predicted versus measured for a combination of several intervals

    # Input:
    # Model is the output from ipls.m, plsmodel.m or plspredict.m
    # no_of_int_lv is the number of PLS components to use for the interval model
    # interval: should be given if ipls model is input, otherwise state [] or omit. Use 0 for global model.
    # y_variable is the number of the y-variable that the plot is made for
    # in the case of only one y-variable simply omit or type 1
    # Copyright, Chemometrics Group - KVL, Copenhagen, Denmark
    # Lars Nørgaard, July 2004
    # plspvsm(Model,no_of_int_lv);

    if not Model:
        print(' ')
        print(' plspvsm(Model,no_of_int_lv,interval,y_variable);')
        print(' ')
        print(' Example:')
        print(' plspvsm(Model,5,10,1);')
        print(' ')
        print(' plspvsm(Model,5);')
        print(' ')
        return

    if Model["type"] not in ['PLS','iPLS','PLSprediction']:
        print(' ')
        print('This function only works with output from ipls.m, plsmodel.m or plspredict.m')
        print(' ')
        return

    if Model["type"] == 'iPLS' and (interval==None and y_variable==None):
        print(' ')
        print('Plotting results from iPLS model: Remember to give interval number as the third parameter')
        print(' ')
        return

    if interval and Model["type"] in ['PLS','PLSprediction']:
        print(' ')
        print('Plotting results from PLS/PLSprediction model: It is not necessary to specify interval')
        print('Use [] or omit if last parameter')
        print(' ')

    if y_variable==None:
        y_variable = 1

    if interval!=None:
        if interval == 0 and Model["type"] == 'iPLS':
            interval = Model["intervals"]+ 1
        elif interval == 0 and Model["type"] == 'PLSprediction':
            interval = Model["CalModel"]["intervals"] + 1

    if (interval==None) and Model["type"] == 'PLSprediction':
        interval = Model["CalModel"]["intervals"] + 1

    plt.rcParams["figure.figsize"] = (16,8)

    ScrLength = 1920
    ScrHight = 1080

    bdwidth = 10

    pos1 = [bdwidth, (0.4 * ScrHight + bdwidth), (ScrLength / 2 - 2 * bdwidth), ScrHight / 1.7 - (70 + bdwidth)]
    pos2 = [pos1[0] + ScrLength / 2, pos1[1], pos1[2], pos1[3]]

    # Position of interval(s)
    #fig1 = plt.figure(figsize=(8,8))
    #ax1 = fig1.add_subplot(111)
    #fig, (ax1, ax2) = plt.subplots(1, 2)

    if Model["type"] == 'iPLS':
        if len(Model["xaxislabels"])==0:
            plt.plot(Model["rawX"].T, 'k')
            #ax1.plot(Model["rawX"].T, 'k')
            plt.xlabel('Variables')
            stvar = Model["allint"][interval-1, 1]
            endvar = Model["allint"][interval-1, 2]
            if interval < Model["allint"].shape[0]:
                titletext = f'Interval number {interval}, variables {stvar}-{endvar}'
            else:
                titletext = f'Global model, variables {stvar}-{endvar}'
        else:
            plt.plot(Model["xaxislabels"].T, Model["rawX"].T, 'k')
            #ax1.plot(Model["xaxislabels"].T, Model["rawX"].T, 'k')
            plt.xlabel('Wavelength')
            stwav = Model["xaxislabels"][:,Model["allint"][interval-1, 1]]
            endwav = Model["xaxislabels"][:,Model["allint"][interval-1, 2]]
            if interval < Model["allint"].shape[0]:
                titletext = f'Interval number {interval}, wavelengths {stwav}-{endwav}'
            else:
                titletext = f'Global model, wavelengths {stwav}-{endwav}'
        ytext = f'Response, raw data [{Model["prepro_method"]} is used in the calculations]'
        plt.ylabel(ytext)

        plt.title(titletext)
        plt.axis('tight')
        actualaxis = plt.axis()
        #plt.hold(True)
        a = Model["allint"][interval-1, 1]
        b = Model["allint"][interval-1, 2]
        #print(f'a is {a} b is {b} c is {Model["xaxislabels"][:,a]} d {Model["xaxislabels"][:,b]}')
        if len(Model["xaxislabels"])==0:
            #h1temp = plt.fill_between(np.arange(a, b + 1), actualaxis[2], actualaxis[2], facecolor=[0.75, 0.75, 0.75])
            #h2temp = plt.fill_between(np.arange(a, b + 1), actualaxis[3], actualaxis[3], facecolor=[0.75, 0.75, 0.75])
            h1temp = plt.fill_between([a, b], [actualaxis[2], actualaxis[2]], facecolor=[0.75, 0.75, 0.75])
            h2temp = plt.fill_between([a, b], [actualaxis[3], actualaxis[3]], facecolor=[0.75, 0.75, 0.75])
            plt.plot(Model["rawX"].T, 'k') # To overlay spectra on area plot
        else:
            #h1temp = plt.fill_between(Model["xaxislabels"][:,np.arange(a,b+1)], actualaxis[2], actualaxis[2], facecolor=[0.75, 0.75, 0.75])
            #h2temp = plt.fill_between(Model["xaxislabels"][:,np.arange(a,b+1)], actualaxis[3], actualaxis[3], facecolor=[0.75, 0.75, 0.75])
            h1temp = plt.fill_between([Model["xaxislabels"][:,a][0],Model["xaxislabels"][:,b][0]], [actualaxis[2], actualaxis[2]], facecolor=[0.75, 0.75, 0.75])
            h2temp = plt.fill_between([Model["xaxislabels"][:,a][0],Model["xaxislabels"][:,b][0]], [actualaxis[3], actualaxis[3]], facecolor=[0.75, 0.75, 0.75])
            plt.plot(Model["xaxislabels"].T, Model["rawX"].T, 'k') # To overlay spectra on area plot
        #plt.hold(False)
        plt.show()


        
    elif Model["type"] == 'PLS':
        if len(Model["xaxislabels"])==0:
            plt.plot(Model["rawX"].T, 'k')
            plt.xlabel('Variables')
            if 'windowsize' in Model:
                stvar = Model["selected_vars"][0]
                endvar = Model["selected_vars"][-1]
                titletext = f'Selected variables [{Model["selected_vars"][0]} to {Model["selected_vars"][-1]}]'
            else:
                stvar = Model["allint"][Model["selected_intervals"], 1]
                endvar = Model["allint"][Model["selected_intervals"], 2]
                titletext = f'Selected intervals [{", ".join(map(str, Model["selected_intervals"]))}]'
        else:
            plt.plot(Model["xaxislabels"].T, Model["rawX"].T, 'k')
            #ax1.plot(Model["xaxislabels"].T, Model["rawX"].T, 'k')
            plt.xlabel('Wavelength')
            if 'windowsize' in Model:
                stwav = Model["xaxislabels"][Model["selected_vars"][0]]
                endwav = Model["xaxislabels"][Model["selected_vars"][-1]]
                titletext = f'Selected variables [{Model["selected_vars"][0]} to {Model["selected_vars"][-1]}]'
            else:
                #print(f'axis {Model["xaxislabels"].shape} ints {Model["allint"]} sel ints {np.array(Model["selected_intervals"])-1}')
                stwav = Model["xaxislabels"][:,Model["allint"][np.array(Model["selected_intervals"])-1, 1]]
                endwav = Model["xaxislabels"][:,Model["allint"][np.array(Model["selected_intervals"])-1, 2]]
                titletext = f'Selected intervals [{", ".join(map(str, Model["selected_intervals"]))}]'
        ytext = f'Response, raw data [{Model["prepro_method"]} is used in the calculations]'
        plt.ylabel(ytext)
        plt.title(titletext)
        plt.axis('tight')
        actualaxis = plt.axis()
        print(f'actual axis is {actualaxis}')
        #plt.hold(True)
        if len(Model["xaxislabels"])==0:
            if 'windowsize' in Model:
                a = Model["selected_vars"][0]
                b = Model["selected_vars"][-1]
            else:
                a = Model["allint"][np.array(Model["selected_intervals"])-1, 1] #[Model["selected_intervals"], 1]
                b = Model["allint"][np.array(Model["selected_intervals"])-1, 2] #[Model["selected_intervals"], 2]
            #print(f'a is {a} and {b}')
            for i in range(len(a)):
                #h1temp = plt.fill_between([a[:,i][0], b[:,i][0]], [actualaxis[2], actualaxis[2]], color=[0.75, 0.75, 0.75])
                #h2temp = plt.fill_between([a[:,i][0], b[:,i][0]], [actualaxis[3], actualaxis[3]], color=[0.75, 0.75, 0.75])
                h1temp = plt.fill_between([a[i], b[i]], [actualaxis[2], actualaxis[2]], color=[0.75, 0.75, 0.75])
                h2temp = plt.fill_between([a[i], b[i]], [actualaxis[3], actualaxis[3]], color=[0.75, 0.75, 0.75])
        else:
            if 'windowsize' in Model:
                a = Model["xaxislabels"][:,Model["selected_vars"][0]]
                b = Model["xaxislabels"][:,Model["selected_vars"][-1]]
            else:
                a = Model["xaxislabels"][:,Model["allint"][np.array(Model["selected_intervals"])-1, 1]]
                b = Model["xaxislabels"][:,Model["allint"][np.array(Model["selected_intervals"])-1, 2]]
            #print(f'a is {a.shape} nnn {a[:,2][0]} and {b[:,2][0]}')
            for i in range(a.shape[1]):
                h1temp = plt.fill_between([a[:,i][0], b[:,i][0]], [actualaxis[2], actualaxis[2]], color=[0.75, 0.75, 0.75])
                h2temp = plt.fill_between([a[:,i][0], b[:,i][0]], [actualaxis[3], actualaxis[3]], color=[0.75, 0.75, 0.75])
        if len(Model["xaxislabels"])==0:
            plt.plot(Model["rawX"].T, 'k')
            #ax1.plot(Model["rawX"].T, 'k')
        else:
            plt.plot(Model["xaxislabels"].T, Model["rawX"].T, 'k')
            #ax1.plot(Model["xaxislabels"].T, Model["rawX"].T, 'k')
        #plt.hold(False)
        #print(f'a2 is {a} and b2 {b}')

    elif Model["type"] == 'PLSprediction':                                                    
        if len(Model["CalModel"]["xaxislabels"])==0:
            plt.plot(Model["CalModel"]["rawX"].T, 'k')
            plt.xlabel('Variables')
            if 'windowsize' in Model["CalModel"]:  # If predModel/oneModel is based on mwModel
                stvar = Model["CalModel"]["selected_vars"][0]
                endvar = Model["CalModel"]["selected_vars"][-1]
                titletext = f'Selected variables [{Model["CalModel"]["selected_vars"][0]} to {Model["CalModel"]["selected_vars"][-1]}]'
            else:
                stvar = Model["CalModel"]["allint"][Model["CalModel"]["selected_intervals"], 1]
                endvar = Model["CalModel"]["allint"][Model["CalModel"]["selected_intervals"], 2]
                titletext = f'Selected intervals [{", ".join(str(i) for i in Model["CalModel"]["selected_intervals"])}]'
        else:
            plt.plot(Model["CalModel"]["xaxislabels"].T, Model["CalModel"]["rawX"].T, 'k')
            plt.xlabel('Wavelength')
            if 'windowsize' in Model["CalModel"]:  # If oneModel is based on mwModel
                stwav = Model["CalModel"]["xaxislabels"][Model["CalModel"]["selected_vars"][0]]
                endwav = Model["CalModel"]["xaxislabels"][Model["CalModel"]["selected_vars"][-1]]
                titletext = f'Selected variables [{Model["CalModel"]["selected_vars"][0]} to {Model["CalModel"]["selected_vars"][-1]}]'
            else:
                stwav = Model["CalModel"]["xaxislabels"][Model["CalModel"]["allint"][Model["CalModel"]["selected_intervals"], 1]]
                endwav = Model["CalModel"]["xaxislabels"][Model["CalModel"]["allint"][Model["CalModel"]["selected_intervals"], 2]]
                titletext = f'Selected intervals [{", ".join(str(i) for i in Model["CalModel"]["selected_intervals"])}]'
        ytext = f'Response, raw data [{Model["CalModel"]["prepro_method"]} is used in the calculations]'
        plt.ylabel(ytext)
        plt.title(titletext)
        plt.axis('tight')
        actualaxis = plt.axis()
        #plt.hold(True)
                                                                            
        
        if len(Model["CalModel"]["xaxislabels"])==0: # If empty
            if hasattr(Model["CalModel"], 'windowsize'): # If oneModel is based on mwModel
                a = Model["CalModel"]["selected_vars"][0]
                b = Model["CalModel"]["selected_vars"][-1]
            else:
                a = Model["CalModel"]["allint"][Model["CalModel"]["selected_intervals"]-1, 1]
                b = Model["CalModel"]["allint"][Model["CalModel"]["selected_intervals"]-1, 2]

            for i in range(len(a)):
                h1temp = plt.fill_between([a[i], b[i]], [actualaxis[2], actualaxis[2]], color=(0.75, 0.75, 0.75)) # Negative areas
                h2temp = plt.fill_between([a[i], b[i]], [actualaxis[3], actualaxis[3]], color=(0.75, 0.75, 0.75))
    
            plt.plot(Model["CalModel"]["rawX"].T, 'k') # To overlay spectra on area plot
        else:
            if hasattr(Model["CalModel"], 'windowsize'): # If oneModel is based on mwModel
                a = Model["CalModel"]["xaxislabels"][Model["CalModel"]["selected_vars"][0]]
                b = Model["CalModel"]["xaxislabels"][Model["CalModel"]["selected_vars"][-1]]
            else:
                a = Model["CalModel"]["xaxislabels"][Model["CalModel"]["allint"][Model["CalModel"]["selected_intervals"]-1, 1]]
                b = Model["CalModel"]["xaxislabels"][Model["CalModel"]["allint"][Model["CalModel"]["selected_intervals"]-1, 2]]

            for i in range(len(a)):
                h1temp = plt.fill_between([a[i], b[i]], [actualaxis[2], actualaxis[2]], color=(0.75, 0.75, 0.75)) # Negative areas
                h2temp = plt.fill_between([a[i], b[i]], [actualaxis[3], actualaxis[3]], color=(0.75, 0.75, 0.75))

            plt.plot(Model["CalModel"]["xaxislabels"].T, np.transpose(Model["CalModel"]["rawX"]), 'k') # To overlay spectra on area plot

        plt.xlabel('Variables' if (Model["CalModel"]["xaxislabels"]==[] or Model["CalModel"]["xaxislabels"]==None) else 'Wavelength')
        plt.ylabel(f'Response, raw data [{Model["CalModel"]["prepro_method"]} is used in the calculations]')
        if len(Model["CalModel"]["xaxislabels"])==0 :
            titletext = f'Selected intervals [{Model["CalModel"]["selected_intervals"]}]'
        else:
            titletext = f'Selected variables [{Model["CalModel"]["selected_vars"][0]} to {Model["CalModel"]["selected_vars"][-1]}]'
        plt.title(titletext)
        plt.axis('tight')
        actualaxis = plt.axis()
    
    plt.show()

                                                                            

    
    #fig, ax = plt.subplots(figsize=pos2)
    # Predicted versus measured for combined intervals
    if Model["type"] == 'iPLS':
        if Model["val_method"].lower() == 'test':
            plotYref = Model["rawY"][Model["segments"]-1, y_variable-1]
            plotYpred = Model["PLSmodel"][interval]["Ypred"][Model["segments"]-1, y_variable-1, no_of_int_lv-1]
            samplelabels = [str(i) for i in Model["segments"]] #np.array_str(Model["segments"])
        else:
            plotYref = Model["rawY"][:, y_variable-1]
            plotYref = plotYref.reshape(-1,1) 
            plotYpred = Model["PLSmodel"][interval-1]["Ypred"][:, y_variable-1, no_of_int_lv-1]
            plotYpred = plotYpred.reshape(-1,1)
            samplelabels = [str(i) for i in range(1, len(plotYref) + 1)] #np.array_str(np.arange(1, len(plotYref) + 1))
        plt.plot(plotYref, plotYpred, 'w')
        for x, y, label in zip(plotYref, plotYpred, samplelabels):
            plt.text(x, y, label)
        a = np.min([plotYref, plotYpred])
        b = np.max([plotYref, plotYpred])
        #ax.set_xlim([a - abs(a) * 0.1, b + abs(b) * 0.1])
        #ax.set_ylim([a - abs(a) * 0.1, b + abs(b) * 0.1])
        plt.axis([a - abs(a)*0.1, b + abs(b)*0.1, a - abs(a)*0.1, b + abs(b)*0.1])
        s = f"{titletext}, with {no_of_int_lv:g} PLS comp. for y-var. no. {y_variable:g}"
        plt.title(s)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        #plt.hold(True)
        plt.plot([a - abs(a) * 0.1, b + abs(b) * 0.1], [a - abs(a) * 0.1, b + abs(b) * 0.1])
        #plt.hold(False)

        r = np.corrcoef(plotYref[:,0], plotYpred[:,0])
        s1 = f"r = {r[0, 1]:.4f}"
        RMSE, Bias = rmbi(plotYref, plotYpred)

        if Model["val_method"].lower() == 'test':
            s2 = f"RMSEP = {RMSE:.4f}"
        else:
            s2 = f"RMSECV = {RMSE:.4f}"
        s3 = f"Bias = {Bias:.4f}"
        plt.text(a + abs(a * 0.08), 1.1 * b - abs(b * 0.05), s1)
        plt.text(a + abs(a * 0.08), 1.1 * b - abs(b * 0.1), s2)
        plt.text(a + abs(a * 0.08), 1.1 * b - abs(b * 0.15), s3)
                                                                            
                                                                            
    elif Model["type"] == 'PLS':
        if Model["val_method"].lower() == 'test':
            plotYref = Model["rawY"][Model["segments"]-1, y_variable-1]
            plotYpred = Model["PLSmodel"][0]["Ypred"][Model["segments"]-1, y_variable-1, no_of_int_lv-1]
            samplelabels = [str(i) for i in Model["segments"]]
        else:
            plotYref = Model["rawY"][:, y_variable-1]
            plotYref = plotYref.reshape(-1,1) 
            plotYpred = Model["PLSmodel"][0]["Ypred"][:, y_variable-1, no_of_int_lv-1]
            plotYpred = plotYpred.reshape(-1,1) 
            #print(f'yref is {Model["rawY"][:, y_variable-1]} and ypred is {Model["PLSmodel"][0]["Ypred"]}')
            samplelabels = [str(i) for i in range(1, len(plotYref) + 1)]

        plt.plot(plotYref, plotYpred, 'w')
        for i, txt in enumerate(samplelabels):
            plt.annotate(txt, (plotYref[i], plotYpred[i]))

        a = min([plotYref.min(), plotYpred.min()])
        b = max([plotYref.max(), plotYpred.max()])
        plt.axis([a - abs(a)*0.1, b + abs(b)*0.1, a - abs(a)*0.1, b + abs(b)*0.1])
        s = f"{titletext}, with {no_of_int_lv} PLS comp. for y-var. no. {y_variable}"
        plt.title(s)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.plot([a - abs(a)*0.1, b + abs(b)*0.1], [a - abs(a)*0.1, b + abs(b)*0.1])

        r = np.corrcoef(plotYref[:,0], plotYpred[:,0])
        s1 = f"r = {r[0,1]:.4f}"
        RMSE, Bias = rmbi(plotYref, plotYpred)

        if Model["val_method"].lower() == 'test':
            s2 = f"RMSEP = {RMSE:.4f}"
        else:
            s2 = f"RMSECV = {RMSE:.4f}"

        s3 = f"Bias = {Bias:.4f}"
        plt.text(a + abs(a * 0.08), 1.1 * b - abs(b * 0.05), s1)
        plt.text(a + abs(a * 0.08), 1.1 * b - abs(b * 0.10), s2)
        plt.text(a + abs(a * 0.08), 1.1 * b - abs(b * 0.15), s3)

    elif Model["type"] == 'PLSprediction':
        plotYref = Model["Yref"][:, y_variable-1]
        plotYpred = Model["Ypred"][:, y_variable-1, no_of_int_lv-1]
    
        plt.plot(plotYref, plotYpred, 'w')
        for i in range(plotYref.shape[0]):
            plt.text(plotYref[i], plotYpred[i], str(i+1))
    
        a = np.min([plotYref, plotYpred])
        b = np.max([plotYref, plotYpred])
        plt.axis([a - abs(a)*0.1, b + abs(b)*0.1, a - abs(a)*0.1, b + abs(b)*0.1])
    
        s = f'{titletext}, with {no_of_int_lv} PLS comp. for y-var. no. {y_variable}'
        plt.title(s)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.plot([a-abs(a)*0.1, b+abs(b)*0.1], [a-abs(a)*0.1, b+abs(b)*0.1])
    
        r = np.corrcoef(plotYref[:,0], plotYpred[:,0])
        s1 = f'r = {r:.4f}'
        RMSE, Bias = rmbi(plotYref, plotYpred)
        s2 = f'RMSEP = {RMSE:.4f}'
        s3 = f'Bias = {Bias:.4f}'
        plt.text(a+abs(a*0.08), 1.1*b-abs(b*0.05), s1)
        plt.text(a+abs(a*0.08), 1.1*b-abs(b*0.10), s2)
        plt.text(a+abs(a*0.08), 1.1*b-abs(b*0.15), s3)
        
    plt.show()