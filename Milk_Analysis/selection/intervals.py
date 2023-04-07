import numpy as np
#from packages import np
def intervals(Model):
    if not Model:
        print('\n intervals(Model);\n')
        return
    
    print('\n')
    if len(Model["xaxislabels"])==0:
        print(' Int.no Start  End  No. vars.')
        number_of_vars = Model["allint"][:, 2] - Model["allint"][:, 1] + 1
        table = np.column_stack((Model["allint"], number_of_vars))
        print(table)
    else:
        print('       Interval   Start var.   End var.   Start wav.   End wav.  Number of vars.')
        number_of_vars = Model["allint"][:, 2] - Model["allint"][:, 1] + 1
        start_wav = Model["xaxislabels"][:,Model["allint"][:, 1]]
        end_wav = Model["xaxislabels"][:,Model["allint"][:, 2]]
        table = np.column_stack((Model["allint"], start_wav.T, end_wav.T, number_of_vars.T))
        #print(table.shape)
        #print(table)
        for i in range(table.shape[0]):
            tabletext = '      {0:2g}         {1:3g}         {2:4g}         {3:4g}           {4:4g}         {5:4g}'.format(table[i,0], table[i,1], table[i,2], table[i,3], table[i,4], table[i,5])
            print(tabletext)