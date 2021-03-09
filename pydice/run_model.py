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

###############################################################################

if __name__ == '__main__':
   

    num_times = 100
    t = np.arange(1, num_times+1)

    start_year = 2000
    final_year = 2500
    years = np.linspace(start_year, final_year, num_times, dtype = np.int32)
    
    p = DiceParams(num_times)
    
    # * Control rate limits
    MIU_up = np.full(num_times, p._limmiu)
    MIU_up[0] = p._miu0
    MIU_up[1:29] = 1.0

    MIU_lo = np.full(num_times,0.01)
    MIU_lo[0] = p._miu0
    MIU_lo[MIU_lo==MIU_up] = 0.99999*MIU_lo[MIU_lo==MIU_up]

    bnds1=[]
    for i in range(num_times):
        bnds1.append((MIU_lo[i],MIU_up[i]))
    # * Control variables
    lag10 = t > num_times - 10
    S_lo = np.full(num_times,1e-1)
    S_lo[lag10] = p._optlrsav
    S_up = np.full(num_times,0.9)
    S_up[lag10] = p._optlrsav
    S_lo[S_lo==S_up] = 0.99999*S_lo[S_lo==S_up]

    bnds2=[]
    for i in range(num_times):
        bnds2.append((S_lo[i],S_up[i]))
        
    # Arbitrary starting values for the control variables:
    S_start = np.full(num_times,0.2)
    S_start[S_start < S_lo] = S_lo[S_start < S_lo]
    S_start[S_start > S_up] = S_lo[S_start > S_up]
    MIU_start = 0.99 * MIU_up
    MIU_start[MIU_start < MIU_lo] = MIU_lo[MIU_start < MIU_lo]
    MIU_start[MIU_start > MIU_up] = MIU_up[MIU_start > MIU_up]

    x_start = np.concatenate([MIU_start,S_start])
    bnds = bnds1 + bnds2

    outputType = 1

    argsv = [-1.0, outputType, num_times, 
             p._tstep, 
             p._al, p._l, p._sigma, p._cumetree, p._forcoth, 
             p._cost1, p._etree, p._scale1, p._scale2, 
             p._ml0, p._mu0, p._mat0, 
             p._a1, p._a2, p._a3, 
             p._c1, p._c3, p._c4,
             p._b11, p._b12, p._b21, p._b22, p._b32, p._b23, p._b33, 
             p._fco22x, p._t2xco2, 
             p._rr, p._gama,
             p._tocean0, p._tatm0, p._elasmu, p._prstp, p._expcost2, 
             p._k0, p._dk, p._pbacktime]
         
    args = tuple(argsv)

    output = simulateDynamics(x_start, *args)        

    dumpState(years, output, "./results/base_case_state_pre_opt.csv")

    outputType = 0
    argsv[1] = outputType
    args = tuple(argsv)
    methodStr = 'SLSQP'

    start = time.time()
    
    results = opt.minimize(objFn, x_start, args, method=methodStr, tol = 1e-10, bounds = tuple(bnds), options={'disp': True})

    end = time.time()
    print("Time Elapsed:", end - start)

    outputType = 1
    argsv[1] = outputType
    args = tuple(argsv)
    
    output = simulateDynamics(results.x, *args)
    dumpState(years, output, "./results/base_case_state_post_opt.csv")
    
    fileName = "./results/base_case_output.pdf" 
    
    plotStateToFile(fileName, years, output, results.x)

    print("Completed.")





