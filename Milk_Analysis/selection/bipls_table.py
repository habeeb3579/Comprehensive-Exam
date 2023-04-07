def biplstable(biModel):
    if biModel is None:
        print(' ')
        print(' biplstable(biModel);')
        print(' ')
        return
    StartSizeRevIntInfo = biModel['RevIntInfo'].shape[0]
    print(f' check sizes {biModel["RevIntInfo"].shape[0]} and {biModel["RevRMSE"].shape[0]}')
    if biModel['RevIntInfo'].shape[0] != biModel['RevRMSE'].shape[0]:
        start_interval_to_remove = biModel['RevRMSE'].shape[0]
        biModel['RevIntInfo'] = biModel['RevIntInfo'][0:start_interval_to_remove]
        biModel['RevRMSE'] = biModel['RevRMSE'][0:start_interval_to_remove]
        biModel['RevVars'] = biModel['RevVars'][0:start_interval_to_remove]
    print(' ')
    print('    Number    Interval    RMSE      Number of Variables')
    print('    ---------------------------------------------------')
    for i in range(biModel['RevIntInfo'].shape[0]):
        tabletext = '      {0:2g}         {1:2g}       {2:0.4f}           {3:3g}'.format(StartSizeRevIntInfo - (i - 1)-2, biModel['RevIntInfo'][i], biModel['RevRMSE'][i], biModel['RevVars'][i])
        print(tabletext)
    print(' ')