import numpy as np
import matplotlib.pyplot as plt
#from packages import np, plt
def iplsplot(Model, labeltype, optimal_lv_global=None, max_yaxis=None, plottype=None):
    """
    iplsplot plots results from iPLS analysis in a bar plot

    Args:
    - Model (the output from ipls.m)
    - labeltype (str): designates whether you want:
        - interval number ('intlabel'),
        - variable number ('varlabel')
        - wavelength number ('wavlabel')
    - optimal_lv_global (int, optional): the number of PLS components chosen for full spectrum model
        - if not given or given by None, the first RMSECV/RMSEP minimum is chosen
    - max_yaxis (float, optional): can be used to control scaling of the iPLS plot
    - plottype (str, optional): 'Cum' (default), 'Cum2' (with the RMSE values of the two preceeding PLS components),
      or 'n' (where is a locked number of PLS components)

    Returns:
    None

    """
    if Model is None:
        print(' ')
        print(' iplsplot(Model,labeltype,optimal_lv_global,max_yaxis,plottype);')
        print(' or')
        print(' iplsplot(Model,labeltype);')
        print(' ')
        print(' Examples:')
        print(' iplsplot(Model,''intlabel'',[],[],''Cum2'');')
        print(' ')
        print(' iplsplot(Model,''intlabel'');')
        print(' ')
        return

    # Error checks
    if not isinstance(Model, dict):
        print('Model input should be a dictionary (output from ipls.m)')
        return

    if Model['type'] != 'iPLS':
        print('The model input is not from an iPLS model')
        return

    plottype_cell = ['Cum', 'Cum2'] + [str(i) for i in range(Model['no_of_lv'])]


    if plottype is None:
        plottype = 'Cum'
        
    if plottype not in plottype_cell:
        print('Invalid plottype')
        return
    
    if labeltype not in {'intlabel', 'varlabel', 'wavlabel'}:
        print('Not legal labeltype, use ''intlabel'',''varlabel'', or ''wavlabel''')
        return

    if Model["intervalsequi"] == 0 and labeltype == 'intlabel':
        print(' ')
        print(' Manually chosen intervals are not correctly plotted with ''intlabel''')
        print(' so please use ''varlabel'' or ''wavlabel'' as labeltype')
        print(' ')
        return

    Xmean = np.mean(Model["rawX"], axis=0) # Mean spectrum
    if np.min(Xmean) < 0:
        Xmean = Xmean + (-np.min(Xmean)) # To make all intensities positive
    n, m = Model["rawX"].shape

    No_Int = Model["intervals"]
    if labeltype == 'intlabel':
        Xtext = 'Interval number'
        Xint = np.column_stack([(Model['allint'][0:No_Int, 0]-0.5).reshape(-1,1), (Model['allint'][0:No_Int, 0]-0.5).reshape(-1,1), (Model['allint'][0:No_Int, 0]+0.5).reshape(-1,1), (Model['allint'][0:No_Int, 0]+0.5).reshape(-1,1)])
        Xint = Xint.T
        #print(f'xint {Xint}')
        NumberofTicksandWhere = np.mean(Xint[1:3, :], axis=0)
        NumberofTicksandWhere = np.array([int(x) for x in NumberofTicksandWhere])
        #print(f'xint {NumberofTicksandWhere}')

    elif labeltype == 'wavlabel':
        if len(Model["xaxislabels"])==0:
            print('You must define wavelength/wavenumber labels')
            return
        Xtext = 'Wavelength/Wavenumber'
        a = Model["allint"][0:No_Int, 1]
        b = Model["allint"][0:No_Int, 2]

        # To reverse wavenumber axis before plotting; will be reversed back when the
        # final plot is made
        NewAxisLabels = Model["xaxislabels"]  # Important; original axislabels are used in the last three lines of the program
        if NewAxisLabels[:,0] > NewAxisLabels[:,1]:
            if NewAxisLabels.shape[0] == 1:
                NewAxisLabels = np.flipud(NewAxisLabels)
            elif NewAxisLabels.shape[1] == 1:
                NewAxisLabels = np.fliplr(NewAxisLabels)

        #print(f'newaxis {NewAxisLabels}')
        Xint = np.column_stack([(NewAxisLabels[:,a]).reshape(-1,1), (NewAxisLabels[:,a]).reshape(-1,1), (NewAxisLabels[:,b]).reshape(-1,1), (NewAxisLabels[:,b]).reshape(-1,1)]).T
        NumberofTicksandWhere = np.append(Xint[1, :], Xint[2, -1])
        #print(f'NoTicks {NumberofTicksandWhere}')

    elif labeltype == 'varlabel':
        Xtext = 'Variable number'
        Xint = np.column_stack((Model["allint"][0:No_Int, 1].reshape(-1,1), Model["allint"][0:No_Int, 1].reshape(-1,1), 
                      Model["allint"][0:No_Int, 2].reshape(-1,1), Model["allint"][0:No_Int, 2].reshape(-1,1)))
        NumberofTicksandWhere = np.append((Xint[1, :], Xint[2, -1]))

    RMSE = []    
    for i in range(Model["intervals"]+1):
        RMSE.append(Model["PLSmodel"][i]["RMSE"])
        #RMSE[i, :] = Model["PLSmodel"][i]["RMSE"]

    RMSE = np.array(RMSE)
    #min_ix = np.zeros(RMSE.shape[0]) 
    #minRMSE = np.zeros(RMSE.shape[0])
    min_ix = [] 
    minRMSE = []
    if plottype in ['Cum', 'Cum2']:
        # [minRMSE,min_ix]=min(RMSE'); % Global minima; could be changed using e.g. F-test or equal

        # First local minima is better; could be changed using e.g. F-test or equal NOT IMPLEMENTED
        # Ones appended to make the search stop if the first local miminum is the last PLSC
        RedRMSE = RMSE[:, 1:] # PLSC0 is excluded in finding the first local minimum
        SignMat = np.hstack((np.sign(np.diff(RedRMSE, axis=1)), np.ones((No_Int+1, 1))))

        #min_ix = np.zeros(RedRMSE.shape[0]) 
        #minRMSE = np.zeros(RedRMSE.shape[0])
        for i in range(RedRMSE.shape[0]):
            for j in range(RedRMSE.shape[1]):
                if SignMat[i, j] == 1:
                    #min_ix[i] = j # Note: PLSC0 is excluded
                    min_ix.append(j)
                    #minRMSE[i] = RedRMSE[i, j] # Note: PLSC0 is excluded
                    minRMSE.append(RedRMSE[i,j])
                    break

    elif plottype in plottype_cell[2:]:
        RedRMSE = RMSE[:, 1:] # PLSC0 is excluded
        #min_ix = np.zeros(RedRMSE.shape[0]) 
        #minRMSE = np.zeros(RedRMSE.shape[0])
        for i in range(RedRMSE.shape[0]):
            min_ix.append(int(plottype))
            #min_ix[i] = int(plottype)
            minRMSE.append(RedRMSE[i, min_ix[i]])
            #minRMSE[i] = RedRMSE[i, min_ix[i]] # Note: PLSC0 is excluded

    min_ix = np.array(min_ix)
    minRMSE = np.array(minRMSE)
    #print(f'minix {min_ix}')
    #print(f'minrmse {minRMSE}')
    if (optimal_lv_global==None and max_yaxis==None and plottype==None):
        optimal_lv_global = min_ix[Model["intervals"]]
    if (max_yaxis!=None or plottype!=None) and (optimal_lv_global==None):
        optimal_lv_global = min_ix[Model["intervals"]]

    plt.rcParams["figure.figsize"] = (16, 8)
    Response = np.column_stack([np.zeros((No_Int,1)), minRMSE[0:No_Int].reshape(-1,1), minRMSE[0:No_Int].reshape(-1,1), np.zeros((No_Int,1))])
    Response = Response.T
    #print(f'response is {Response}')
    ResponseMinusOnePC = np.zeros_like(Response)
    #minRMSEminusOne = np.array(No_Int)
    minRMSEminusOne = []

    for i in range(No_Int):
        if min_ix[i] == 0:
            minRMSEminusOne.append(np.nan)
            #minRMSEminusOne[i] = np.nan
        else:
            minRMSEminusOne.append(RMSE[i, min_ix[i]])
            #minRMSEminusOne[i] = RMSE[i, min_ix[i]]
    minRMSEminusOne = np.array(minRMSEminusOne)
    ResponseMinusOnePC[1, :] = minRMSEminusOne
    ResponseMinusOnePC[2, :] = minRMSEminusOne
    #print(f'minrmse {minRMSEminusOne}')
    #print(f'resmintwo {ResponseMinusOnePC}')
    
    ResponseMinusTwoPC = np.zeros_like(Response)
    #minRMSEminusTwo = np.array(No_Int)
    minRMSEminusTwo = []
    for i in range(No_Int):
        if min_ix[i] <= 1:
            minRMSEminusTwo.append(np.nan)
            #minRMSEminusTwo[i]=np.nan
        else:
            minRMSEminusTwo.append(RMSE[i, min_ix[i]-1])
            #minRMSEminusTwo[i] = RMSE[i, min_ix[i]-1]

    minRMSEminusTwo = np.array(minRMSEminusTwo)
    ResponseMinusTwoPC[1, :] = minRMSEminusTwo
    ResponseMinusTwoPC[2, :] = minRMSEminusTwo
    #print(f'resmintwo {ResponseMinusTwoPC}')
    #print(f'plottype {[x for x in plottype_cell if x!=plottype_cell[1]]}')
    #print(f'plottype {plottype}')
    #print(f'xaxis {Model["xaxislabels"].shape}')
    #print(f'xint is {np.flipud(Xint.T.ravel())} res2pc {ResponseMinusTwoPC.T.ravel()} res1pc {ResponseMinusOnePC.T.ravel()}')
    #print(f'xint is {np.flipud(Xint.T.ravel())} resint {Response.T.ravel()} res1pc {ResponseMinusOnePC.T.ravel()}')
    # Cumulated plots
    if (optimal_lv_global!=None and max_yaxis!=None) and plottype=='Cum2':
        if labeltype=='wavlabel' and Model["xaxislabels"][:,0]>Model["xaxislabels"][:,1]:
            plt.fill_between(np.flipud(Xint.T.ravel()),ResponseMinusTwoPC.T.ravel(),ResponseMinusOnePC.T.ravel(),color='w')
            plt.fill_between(np.flipud(Xint.T.ravel()),Response.T.ravel(),ResponseMinusOnePC.T.ravel(),color=[0.75,0.75,0.75])
        else:
            plt.fill_between(Xint.T.ravel(),ResponseMinusTwoPC.T.ravel(),ResponseMinusOnePC.T.ravel(),color='w')
            plt.fill_between(Xint.T.ravel(),Response.T.ravel(),ResponseMinusOnePC.T.ravel(),color=[0.75,0.75,0.75])
    #ids = [x for x in plottype_cell if x!=plottype_cell[1]]
    #elif plottype in plottype_cell(ids):
    elif plottype in [x for x in plottype_cell if x!=plottype_cell[1]]:
        if labeltype=='wavlabel' and Model["xaxislabels"][:,0]>Model["xaxislabels"][:,1]:
            plt.fill_between(np.flipud(Xint.T.ravel()),Response.T.ravel(),color=[0.75,0.75,0.75])
        else:
            plt.fill_between(Xint.T.ravel(),Response.T.ravel(),color=[0.75,0.75,0.75]) # Note: substitute [0.75 0.75 0.75]'c' for cyan
    else:
        pass

    if Model["val_method"]=='test':
        plottitle = 'Dotted line is RMSEP (%g LV''s) for global model / Italic numbers are optimal LVs in interval model' % optimal_lv_global
        plt.ylabel('RMSEP',fontsize=10)
    elif Model["val_method"]=='none':
        plottitle = 'Dotted line is RMSEC (%g LV''s) for global model / Italic numbers are optimal LVs in interval model' % optimal_lv_global
        plt.ylabel('RMSEC',fontsize=10)
    else:
        plottitle = 'Dotted line is RMSECV (%g LV''s) for global model / Italic numbers are optimal LVs in interval model' % optimal_lv_global
        plt.ylabel('RMSECV',fontsize=10)
    plt.title(plottitle,fontsize=10,fontweight='bold')
    plt.xlabel(Xtext)

    #plt.show()
    
    #plt.hold(True)
    plt.axis('tight')
    plt.axhline(RMSE[No_Int,optimal_lv_global], linestyle=':', color='k')
    actualaxis = plt.axis()
    if (max_yaxis is not None):
        plt.axis([actualaxis[0], actualaxis[1], actualaxis[2], max_yaxis])
        actualaxis[3] = max_yaxis
    Xaxis = np.linspace(actualaxis[0], actualaxis[1], m)
    #print(f'xasis {Xaxis}')
    if (labeltype == 'wavlabel') and (Model["xaxislabels"][:,0] > Model["xaxislabels"][:,1]):
        plt.plot(np.flipud(Xaxis), Xmean / max(Xmean) * actualaxis[3], '-k')
    else:
        plt.plot(Xaxis, Xmean / max(Xmean) * actualaxis[3], '-k')
    plt.xticks(NumberofTicksandWhere)
    #print(f'min ix {Model["intervals"]}')
    #print(f'minix {min_ix.shape}')
    for i in range(Model["intervals"]):
        #print(f'min ix {min_ix[Model["intervals"]]}')
        if (labeltype == 'wavlabel') and (Model["xaxislabels"][:,0] > Model["xaxislabels"][:,1]):
            plt.text(np.mean(Xint[1:3,i]), 0.03*(actualaxis[3]-actualaxis[2])+actualaxis[2], str(min_ix[Model["intervals"]-(i+1-1)-1]), color='k', fontstyle='italic')
        else:
            plt.text(np.mean(Xint[1:3,i]), 0.03*(actualaxis[3]-actualaxis[2])+actualaxis[2], str(min_ix[i]), color='k', fontstyle='italic')
#plt.hold(False)
    if (labeltype == 'wavlabel') and (Model["xaxislabels"][:,0] > Model["xaxislabels"][:,1]):
        plt.gca().invert_xaxis()
        
    plt.show()
        
def horzline(ordinate, linetype_color):
    n, m = ordinate.shape
    V = plt.axis()
    #if plt.ishold():
    plt.plot([V[0], V[1]], [ordinate, ordinate], linetype_color)
    #else:
    #    plt.hold(True)
    #    plt.plot([V[0], V[1]], [ordinate, ordinate], linetype_color)
    #    plt.hold(False)