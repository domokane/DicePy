# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from dice_params import DiceParams
from dice_dynamics import simulateDynamics, dumpState
import pandas as pd

###############################################################################
# Attempt to reconcile with Matlab code from RR
###############################################################################

if __name__ == '__main__':

    num_times = 50
    tstep = 5.0

    t = np.arange(1, num_times+1)

    p = DiceParams(num_times, tstep)
    outputType = 1

    start_year = 2015
    final_year = start_year + p._tstep * num_times
    years = np.linspace(start_year, final_year, num_times, dtype=np.int32)

    argsv = [-1.0, outputType, num_times,
             p._tstep,
             p._al, p._l, p._sigma, p._cumetree, p._forcoth,
             p._cost1, p._etree, p._scale1, p._scale2,
             p._ml0, p._mu0, p._mat0, p._cca0,
             p._a1, p._a2, p._a3,
             p._c1, p._c3, p._c4,
             p._b11, p._b12, p._b21, p._b22, p._b32, p._b23, p._b33,
             p._fco22x, p._t2xco2,
             p._rr, p._gama,
             p._tocean0, p._tatm0, p._elasmu, p._prstp, p._expcost2,
             p._k0, p._dk, p._pbacktime]

    args = tuple(argsv)

    scenariosDF = pd.read_csv("./Reference_Objective_Scenarios.csv")
    scenariosDF.columns = ['Sav1', 'Miu1', 'Sav2', 'Miu2', 'Sav3', 'Miu3']

    S_start = scenariosDF['Sav1'].values
    MIU_start = scenariosDF['Miu1'].values

    # Arbitrary starting values for the control variables:
#    S_start = np.full(num_times,0.2596)
#    MIU_start = np.full(num_times,  0.03)
    x_start = np.concatenate([MIU_start, S_start])

    output = simulateDynamics(x_start, *args)

    dumpState(years, output, "./results/scenarioOutput.csv")

    print("Completed.")
