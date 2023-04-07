import matplotlib.pyplot as plt
import numpy as np


def plsrmse(Model, interval=None):
    """
    plsrmse plots RMSECV/RMSEP/RMSEC as a function of the number of PLS components

    Inputs:
    Model: the output from ipls.m
    interval: optional for selecting RMSECV/RMSEP/RMSEC for one interval or global model (use 0)

    Copyright:
    Chemometrics Group - KVL, Copenhagen, Denmark
    Lars Nørgaard, July 2004

    plsrmse(Model, interval)
    """

    if not Model:
        print("\nplsrmse(Model, interval); or plsrmse(Model); \n")
        return

    if Model['type'] not in ['PLS', 'iPLS', 'PLSprediction']:
        print("\nThis function only works with output from ipls.m, plsmodel.m or plspredict.m\n")
        return

    if Model['type'] == 'PLS':
        # When output is from plsmodel, i.e., a single model is calculated
        #print(f'range is {list(range(Model["no_of_lv"]+1))}')
        #print(f'pls models {Model["PLSmodel"]}')
        #print(f'pls models {Model["selected_intervals"]}')
        #print(f'ints {Model["intervals"]}')
        #print(f'models is {Model["PLSmodel"][Model["intervals"]-1]["RMSE"]}')
        plt.bar(list(range(Model['no_of_lv']+1)), Model['PLSmodel'][Model['intervals']-1]['RMSE'])
        plt.axis('tight')
        actualaxis = plt.axis()
        plt.axis([actualaxis[0], actualaxis[1], 0, actualaxis[3]*1.1])  # *1.1 to give some space in the plot
        if Model['val_method'] == 'none':
            figtitle = f"RMSEC (fit) versus PLS components for model on interval: {Model['selected_intervals']}"
        elif Model['val_method'] == 'test':
            figtitle = f"RMSEP (dependent) versus PLS components for model on interval: {Model['selected_intervals']}"
        else:
            figtitle = f"RMSECV versus PLS components for model on interval: {Model['selected_intervals']}"
        plt.title(figtitle)
        plt.xlabel('Number of PLS components')
        if Model['val_method'] == 'none':
            plt.ylabel('RMSEC')
        elif Model['val_method'] == 'test':
            plt.ylabel('RMSEP')
        else:
            plt.ylabel('RMSECV')
        

    elif Model['type'] == 'iPLS':
        if interval is not None:
            if interval == 0:
                interval = Model['intervals'] + 1
                if Model['val_method'] == 'none':
                    figtitle = "RMSEC (fit) versus PLS components for global model"
                elif Model['val_method'] == 'test':
                    figtitle = "RMSEP (dependent) versus PLS components for global model"
                else:
                    figtitle = "RMSECV versus PLS components for global model"
            else:
                if Model['val_method'] == 'none':
                    figtitle = f"RMSEC (fit) versus PLS components for model on interval {interval}"
                elif Model['val_method'] == 'test':
                    figtitle = f"RMSEP (dependent) versus PLS components for model on interval {interval}"
                else:
                    figtitle = f"RMSECV versus PLS components for model on interval {interval}"
            plt.bar(list(range(Model['no_of_lv']+1)), Model['PLSmodel'][interval-1]['RMSE'])
            #plt.bar(list(range(Model['no_of_lv']+1)), Model['PLSmodel'][Model['intervals']-1]['RMSE'])
            plt.axis('tight')
            actualaxis = plt.axis()
            plt.axis([actualaxis[0], actualaxis[1], 0, actualaxis[3]*1.1])  # *1.
            plt.title(figtitle)
            plt.xlabel('Number of PLS components')
            if Model['val_method'] == 'none':
                plt.ylabel('RMSEC')
            elif Model['val_method'] == 'test':
                plt.ylabel('RMSEP')
            else:
                plt.ylabel('RMSECV')
        elif interval is None:
            RMSE = []
            for i in range(Model["intervals"] + 1):
                #RMSE[i, :] = Model["PLSmodel"][i]["RMSE"]
                RMSE.append(Model["PLSmodel"][i]["RMSE"])
                
            RMSE = np.array(RMSE)
    
            #plt.plot(range(Model["no_of_lv"] + 1), RMSE[0:Model["intervals"] + 1, :].T)
            plt.plot(range(Model["no_of_lv"] + 1), RMSE[np.arange(Model["intervals"])])
            actualaxis = plt.axis()
            plt.axis([actualaxis[0], actualaxis[1], 0, actualaxis[3]])

            if Model["val_method"] == 'none':
                figtitle = 'RMSEC (fit) versus PLS components for all interval models, -o- is RMSEC for the global model'
            elif Model["val_method"] == 'test':
                figtitle = 'RMSEP (dependent) versus PLS components for all interval models, -o- is RMSEP for the global model'
            else:
                figtitle = 'RMSECV versus PLS components for all interval models, -o- is RMSECV for the global model'
    
            plt.title(figtitle)
            plt.xlabel('Number of PLS components')

            if Model["val_method"] == 'none':
                plt.ylabel('RMSEC')
            elif Model["val_method"] == 'test':
                plt.ylabel('RMSEP')
            else:
                plt.ylabel('RMSECV')

            plt.plot(range(Model["no_of_lv"] + 1), RMSE[Model["intervals"]], 'og')
            plt.plot(range(Model["no_of_lv"] + 1), RMSE[Model["intervals"]], '-g')
            #plt.show()

    elif Model["type"] == 'PLSprediction':
        plt.bar(range(Model["no_of_lv"]+1), Model["RMSE"])
        plt.axis('tight')
        actualaxis = plt.axis()
        plt.axis([actualaxis[0], actualaxis[1], 0, actualaxis[3]*1.1])
        figtitle = f"RMSEP (independent) versus PLS components for model on interval: {Model['CalModel']['selected_intervals']}"
        plt.title(figtitle)
        plt.xlabel('Number of PLS components')
        plt.ylabel('RMSEP')
    plt.show()