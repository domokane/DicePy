# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import time

import scipy.optimize as opt

from dice_params import DiceParams
from dice_dynamics import objFn, simulateDynamics, plotStateToFile, dumpState

import matplotlib.pyplot as plt

###############################################################################

if __name__ == '__main__':

    num_times = 100
    tstep = 5.0

    t = np.arange(1, num_times+1)

    p = DiceParams(num_times, tstep)
    outputType = 1

    start_year = 2015
    final_year = start_year + p._tstep * num_times
    years = np.linspace(start_year, final_year, num_times+1, dtype=np.int32)

    scc_index = 0
    c_bump = 0.0
    e_bump = 0.0
    
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
             p._k0, p._dk, p._pbacktime,
             scc_index, e_bump, c_bump]

    args = tuple(argsv)

    # Arbitrary starting values for the control variables:
    S_start = np.full(num_times, 0.2596)
    MIU_start = np.full(num_times,  0.03)
    x_start = np.concatenate([MIU_start, S_start])

    output = simulateDynamics(x_start, *args)

#    dumpState(years, output, "./results/base_case_state_pre_opt.csv")

    ###########################################################################
    # OPTIMISATION
    ###########################################################################
    # BOUNDS

    # * Control emissions rate limits upper limits
    MIU_up = np.full(num_times, p._limmiu)
    MIU_up[0] = p._miu0  # The first value is not optimised but held fixed
    MIU_up[1:29] = 1.0

    # * Control emissions rate limits lower limits
    MIU_lo = np.full(num_times, 0.01)
    MIU_lo[0] = p._miu0
    MIU_lo[MIU_lo == MIU_up] = 0.99999*MIU_lo[MIU_lo == MIU_up]

    bnds1 = []
    for i in range(num_times):
        bnds1.append((MIU_lo[i], MIU_up[i]))

    # Lower and upper limits on the savings rates
    lag10 = t > num_times - 10
    S_lo = np.full(num_times, 1e-1)
    S_lo[lag10] = p._optlrsav

    S_up = np.full(num_times, 0.9)
    S_up[lag10] = p._optlrsav

    S_lo[S_lo == S_up] = 0.99999*S_lo[S_lo == S_up]

    bnds2 = []
    for i in range(num_times):
        bnds2.append((S_lo[i], S_up[i]))

    ###########################################################################

    # Arbitrary starting values for the control variables:
    S_start = np.full(num_times, 0.2)
    S_start[S_start < S_lo] = S_lo[S_start < S_lo]
    S_start[S_start > S_up] = S_lo[S_start > S_up]
    MIU_start = 0.99 * MIU_up
    MIU_start[MIU_start < MIU_lo] = MIU_lo[MIU_start < MIU_lo]
    MIU_start[MIU_start > MIU_up] = MIU_up[MIU_start > MIU_up]

    ###########################################################################

    x_start = np.concatenate([MIU_start, S_start])
    bnds = bnds1 + bnds2

    outputType = 0
    argsv[1] = outputType
    args = tuple(argsv)
    methodStr = 'SLSQP'

    start = time.time()
    for i in range(1, 100):
        x = objFn(x_start, *args)
        pass

    end = time.time()
    print("Speed test Time Elapsed:", end - start)
    tol = 1e-14

    start = time.time()
    results = opt.minimize(objFn, x_start, args, method=methodStr,
                           options={'ftol': tol, 'eps': 1e-7, 'disp': True},
                           bounds=tuple(bnds))
    
    if 1 == 0:
        # This removes some of the oscillations in the Saving Rates.
        methodStr = 'POWELL'
        results = opt.minimize(objFn, x_start, args, method=methodStr,
                               tol=1e-12,
                               options={'disp': True}, bounds=tuple(bnds))

    x_opt = results.x
    w_0 = results.fun

    end = time.time()
    print("Time Elapsed:", end - start)

    ###########################################################################

    outputType = 0 # Welfare only at [0,0]
    argsv[0] = -1.0 # sign is not negative
    argsv[1] = outputType
    scc_vector = []
    tol = 1e-14
    4517.31467991

    for i in range(1, 101): # 102):

        output_m = simulateDynamics(results.x, *args)

        e_bump = 0.01
        c_bump = e_bump

        argsv[-3] = i
        argsv[-2] = e_bump
        argsv[-1] = 0.0        
        args = tuple(argsv)

        results = opt.minimize(objFn, x_start, args, method=methodStr,
                           options={'ftol': tol, 'eps': 1e-7, 'disp': False},
                           bounds=tuple(bnds))

        w_e_x = results.x
        w_e = results.fun
        
        argsv[-3] = i
        argsv[-2] = 0.0
        argsv[-1] = c_bump
        args = tuple(argsv)

        results = opt.minimize(objFn, x_start, args, method=methodStr,
                           options={'ftol': tol, 'eps': 1e-7, 'disp': False},
                           bounds=tuple(bnds))

        w_c_x = results.x
        w_c = results.fun
        
        dwde = (w_e - w_0) / e_bump
        dwdc = (w_c - w_0) / c_bump

 #       for j in range(0, len(w_c_x)):
 #           print(j, w_c_x[j], w_e_x[j], w_c_x[j] - w_e_x[j])
        
        scc = -1000.0 * dwde / (0.000001 + dwdc)
        scc_vector.append(scc)
        print("Period:", i, "Year:", years[i-1], "DWDE", dwde, "DWDC", dwdc, "SCC:", scc) 

    plt.plot(years, scc_vector);
    plt.xlabel("Year")
    plt.ylabel("SCC in $/tonne of Carbon")
    plt.title("Social Cost of Carbon")

    ###########################################################################
    
    outputType = 1
    argsv[1] = outputType
    args = tuple(argsv)

    print("Dumping graphs to file.")
    output = simulateDynamics(results.x, *args)
    dumpState(years, output, "./results/base_case_state_post_opt.csv")

    fileName = "./results/base_case_output.pdf"
#    plotStateToFile(fileName, years, output, results.x)

    print("Completed.")
