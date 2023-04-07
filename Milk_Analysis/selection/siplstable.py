def siplstable(siModel):
    # siPLStable lists optimal interval combinations, corresponding RMSEs and PLSC

    # Input:
    # siModel is the output from sipls.m

    # Copyright, Chemometrics Group - KVL, Copenhagen, Denmark
    # Lars Nørgaard, July 2004
    # siplstable(siModel);

    if not siModel:
        print("\nsiplstable(siModel);\n")
        return

    print(f"\n    Original number of intervals is {siModel['intervals']}\n")

    if len(siModel["IntComb"][0]) == 2:
        print("    PLS comp.  Intervals    RMSE")
        print("    ------------------------------")
    elif len(siModel["IntComb"][0]) == 3:
        print("    PLS comp.     Intervals      RMSE")
        print("    -----------------------------------")
    elif len(siModel["IntComb"][0]) == 4:
        print("    PLS comp.       Intervals         RMSE")
        print("    ----------------------------------------")

    for i in range(len(siModel["minRMSE"][:10])):
        if len(siModel["IntComb"][0]) == 2:
            tabletext = f'      {siModel["minPLSC"][i]:2g}       [{siModel["minComb"][i][0]:2g}   {siModel["minComb"][i][1]:2g}]    {siModel["minRMSE"][i]:0.4g}'
            print(tabletext)
        elif len(siModel["IntComb"][0]) == 3:
            tabletext = f'      {siModel["minPLSC"][i]:2g}       [{siModel["minComb"][i][0]:2g}   {siModel["minComb"][i][1]:2g}   {siModel["minComb"][i][2]:2g}]    {siModel["minRMSE"][i]:0.4g}'
            print(tabletext)
        elif len(siModel["IntComb"][0]) == 4:
            tabletext = f'      {siModel["minPLSC"][i]:2g}       [{siModel["minComb"][i][0]:2g}   {siModel["minComb"][i][1]:2g}   {siModel["minComb"][i][2]:2g}   {siModel["minComb"][i][3]:2g}]    {siModel["minRMSE"][i]:0.4g}'
            print(tabletext)

    print()