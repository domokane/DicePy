# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 23:20:04 2021

@author: Dominic
"""

import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

import csv

###############################################################################
# All arrays have been shifted to have length numtimes + 1 and to start at 
# 1 in line with matlab code
###############################################################################
# Due to my essential wish to use the same code for the optimiser and to get 
# the state info I have created two functions, one wrapped inside the other.
# The optimiser call fOBJ which returns a float for the optimiser. However 
# internally this calls the simulateDynamics function which can return either 
# the single value (utility to be maximised) or the state information.
###############################################################################

@njit(cache=True, fastmath=True)
def objFn(x, *args):
    """ This is the pass-through function that returns a single float value of
    the objective function for the benefit of the optimisation algorithm. """

    out = simulateDynamics(x, *args)    
    return out[0,0]

############################################################################### 
   
@njit(cache=True, fastmath=True)
def simulateDynamics(x, sign, outputType, num_times, 
                     tstep, al, l, sigma, cumetree, forcoth,
                     cost1, etree, 
                     scale1, scale2, 
                     ml0, mu0, mat0, 
                     a1, a2, a3, 
                     c1, c3, c4,
                     b11, b12, b21, b22, b32, b23, b33, 
                     fco22x, t2xco2, rr, gama,
                     tocean0, tatm0, elasmu, prstp, expcost2, 
                     k0, dk, pbacktime):
    """ This is the simulation of the DICE 2016 model dynamics. It is optimised
    for speed. For this reason I have avoided the use of classes. """

    LOG2 = np.log(2)
    L = l # NORDHAUS RENAMES IT TO UPPER CASE IN EQUATIONS
    MILLE = 1000.0
    
    # We take care to ensure that the indexing starts at 1 to allow comparison
    # with matlab
    MIUopt = np.zeros(num_times+1)
    Sopt = np.zeros(num_times+1)

    ###########################################################################
    # Set the optimisation variables
    ###########################################################################
    
    for i in range(1, num_times+1):        
        MIUopt[i] = x[i-1]
        Sopt[i] = x[num_times + i-1]
    
    ###########################################################################
    
    FORC = np.zeros(num_times+1)
    TATM = np.zeros(num_times+1)
    TOCEAN = np.zeros(num_times+1)
    MAT = np.zeros(num_times+1)
    MU = np.zeros(num_times+1)
    ML = np.zeros(num_times+1)
    E = np.zeros(num_times+1)
    EIND = np.zeros(num_times+1)
    C = np.zeros(num_times+1)
    K = np.zeros(num_times+1)
    CPC = np.zeros(num_times+1)
    I = np.zeros(num_times+1)
    RI = np.zeros(num_times+1)
    Y = np.zeros(num_times+1)
    YGROSS = np.zeros(num_times+1)
    YNET = np.zeros(num_times+1)
    DAMAGES = np.zeros(num_times+1)
    DAMFRAC = np.zeros(num_times+1)
    ABATECOST = np.zeros(num_times+1)
    MCABATE = np.zeros(num_times+1)
    CCA = np.zeros(num_times+1)
    CCATOT = np.zeros(num_times+1)
    PERIODU = np.zeros(num_times+1)
    CPRICE = np.zeros(num_times+1)
    CEMUTOTPER = np.zeros(num_times+1)

    # Fixed initial values        
    MAT[1] = mat0
    ML[1] = ml0
    MU[1] = mu0
    TATM[1] = tatm0
    TOCEAN[1] = tocean0
    K[1] = k0

    YGROSS[1] = al[1] * ((L[1]/MILLE)**(1.0-gama)) * K[1]**gama
    EIND[1] = sigma[1] * YGROSS[1] * (1.0 - MIUopt[1])
    E[1] = EIND[1] + etree[1]
    CCA[1] = 0.0 # DOES NOT START TILL PERIOD 2
    CCATOT[1] = CCA[1] + cumetree[1]
    FORC[1] = fco22x * np.log(MAT[1]/588.000)/LOG2 + forcoth[1]
    DAMFRAC[1] = a1*TATM[1] + a2*TATM[1]**a3
    DAMAGES[1] = YGROSS[1] * DAMFRAC[1]
    ABATECOST[1] = YGROSS[1] * cost1[1] * MIUopt[1]**expcost2
    MCABATE[1] = pbacktime[1] * MIUopt[1]**(expcost2-1)
    CPRICE[1] = pbacktime[1] * (MIUopt[1])**(expcost2-1)

    YNET[1] = YGROSS[1] * (1.0 - DAMFRAC[1])
    Y[1] = YNET[1] - ABATECOST[1]
    I[1] = Sopt[1] * Y[1] 
    C[1] = Y[1] - I[1]
    CPC[1] = MILLE * C[1] / L[1]
    # RI[T] is set at end

    PERIODU[1] = ((C[1]*MILLE/L[1])**(1.0-elasmu)-1.0) / (1.0 - elasmu) - 1.0
    CEMUTOTPER[1] = PERIODU[1] * L[1] * rr[1]

    for i in range(2, num_times+1):

        # Depend on t-1
        CCA[i] = CCA[i-1] + EIND[i-1] * 5.0 / 3.666
        MAT[i] = MAT[i-1] * b11 + MU[i-1] * b21 + E[i-1] * 5.0 / 3.666
        ML[i] = ML[i-1] * b33  + MU[i-1] * b23
        MU[i] = MAT[i-1] * b12 + MU[i-1] * b22 + ML[i-1]*b32
        TOCEAN[i] = TOCEAN[i-1] + c4 * (TATM[i-1] - TOCEAN[i-1])

        CCATOT[i] = CCA[i] + cumetree[i]

        K[i] = (1.0-dk)**tstep * K[i-1] + tstep * I[i-1]
        YGROSS[i] = al[i] * ((L[i]/MILLE)**(1.0 - gama)) * K[i]**gama
        EIND[i] = sigma[i] * YGROSS[i] * (1.0 - MIUopt[i])
        E[i] = EIND[i] + etree[i]

        FORC[i] = fco22x * np.log(MAT[i]/588.000)/LOG2 + forcoth[i]
        TATM[i] = TATM[i-1] + c1 * (FORC[i] - (fco22x/t2xco2) * TATM[i-1] - c3 * (TATM[i-1] - TOCEAN[i-1]))

        DAMFRAC[i] = a1 * TATM[i] + a2*TATM[i]**a3
        DAMAGES[i] = YGROSS[i] * DAMFRAC[i]

        ABATECOST[i] = YGROSS[i] * cost1[i] * MIUopt[i]**expcost2
        MCABATE[i] = pbacktime[i] * MIUopt[i]**(expcost2-1)
        CPRICE[i] = pbacktime[i] * (MIUopt[i])**(expcost2-1)
        YNET[i] = YGROSS[i] * (1.0 - DAMFRAC[i])
        Y[i] = YNET[i] - ABATECOST[i]
        I[i] = Sopt[i] * Y[i] 
        C[i] = Y[i] - I[i]
        CPC[i] = MILLE * C[i] / L[i]
        PERIODU[i] = ((C[i]*MILLE/L[i])**(1.0-elasmu) - 1.0) / (1.0 - elasmu) - 1.0
        CEMUTOTPER[i] = PERIODU[i] * L[i] * rr[i]

    for i in range(1, num_times):
        RI[i] = (1.0 + prstp) * (CPC[i+1]/CPC[i])**(elasmu/tstep) - 1.0

    RI[-1] = 0.0

    output = np.zeros((num_times, 25))

    if outputType == 0: 

        resUtility = tstep * scale1 * np.sum(CEMUTOTPER) + scale2
        resUtility *= sign    
        output[0,0] = resUtility    
        return output
        
    elif outputType == 1:

        for iTime in range(1, num_times+1):

            col = 0
            jTime = iTime - 1
            output[jTime, col] = K[iTime]; col += 1 # 0
            output[jTime, col] = YGROSS[iTime]; col += 1 # 1
            output[jTime, col] = EIND[iTime]; col += 1 # 2
            output[jTime, col] = E[iTime]; col += 1 # 3
            output[jTime, col] = CCA[iTime]; col += 1 # 4
            output[jTime, col] = CCATOT[iTime]; col += 1 # 5
            output[jTime, col] = MAT[iTime]; col += 1 # 6
            output[jTime, col] = ML[iTime]; col += 1 # 7
            output[jTime, col] = MU[iTime]; col += 1 # 8
            output[jTime, col] = FORC[iTime]; col += 1 # 9
            output[jTime, col] = TATM[iTime]; col += 1 # 10
            output[jTime, col] = TOCEAN[iTime]; col += 1 # 11
            output[jTime, col] = DAMFRAC[iTime]; col += 1 # 12
            output[jTime, col] = DAMAGES[iTime]; col += 1 # 13
            output[jTime, col] = ABATECOST[iTime]; col += 1 # 14
            output[jTime, col] = MCABATE[iTime]; col += 1 # 15
            output[jTime, col] = CPRICE[iTime]; col += 1 # 16
            output[jTime, col] = YNET[iTime]; col += 1 # 17
            output[jTime, col] = Y[iTime]; col += 1 # 18
            output[jTime, col] = I[iTime]; col += 1 # 19
            output[jTime, col] = C[iTime]; col += 1 # 20
            output[jTime, col] = CPC[iTime]; col += 1 # 21
            output[jTime, col] = RI[iTime]; col += 1 # 22
            output[jTime, col] = PERIODU[iTime]; col += 1 # 23
            output[jTime, col] = CEMUTOTPER[iTime]; col += 1 # 24

        return output
    
    else:
        raise Exception("Unknown output type.") 

    return output
        
###############################################################################

def dumpState(years, output, filename):
     
     f = open(filename, mode = "w", newline='')
     writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

     header = []
     header.append("IPERIOD")
     header.append("K")
     header.append("YGROSS")
     header.append("EIND")
     header.append("E")
     header.append("CCA")
     header.append("CCATOT")
     header.append("MAT")
     header.append("ML")
     header.append("MU")
     header.append("FORC")
     header.append("TATM")
     header.append("TOCEAN")
     header.append("DAMFRAC")
     header.append("DAMAGES")
     header.append("ABATECOST")
     header.append("MCABATE")
     header.append("CPRICE")
     header.append("YNET")
     header.append("Y")
     header.append("I")
     header.append("C")
     header.append("CPC")
     header.append("RI")
     header.append("PERIODU")
     header.append("CEMUTOTPER")
     writer.writerow(header)
     
     num_rows = output.shape[0]
     num_cols = len(header)

     for iTime in range(1, num_rows):
         row = [iTime]
         for iCol in range(0, num_cols-1):
            row.append(output[iTime, iCol])
         writer.writerow(row)

     f.close()
        
###############################################################################

def plotFigure(x, y, xlabel, ylabel, title):

    mpl.style.use('ggplot')
    fig = plt.figure(figsize=(8,6), dpi=72, facecolor="white")
    plt.plot(x, y)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    fig.tight_layout()
    return fig
    
###############################################################################
    
def plotStateToFile(fileName, years, output, x):

    num_times = output.shape[0]
    output = np.transpose(output)

    pp = PdfPages(fileName)

    TATM = output[10]
    title = 'Increase temperature of the atmosphere (TATM)'
    xlabel = 'Years'
    ylabel = 'Degrees C from 1900'
    fig = plotFigure(years, TATM, xlabel, ylabel, title)
    pp.savefig(fig)

    TOCEAN = output[11]
    xlabel = 'Years'
    title='Increase temperature of the ocean (TOCEAN)'
    ylabel= 'Degrees C from 1900'
    fig = plotFigure(years, TOCEAN, xlabel, ylabel, title)
    pp.savefig(fig)

    MU = output[8]    
    xlabel = 'Years'
    title='Carbon concentration increase in shallow oceans (MU)'
    ylabel = 'GtC from 1750'
    fig = plotFigure(years, MU, xlabel, ylabel, title)
    pp.savefig(fig)

    ML = output[7]       
    xlabel = 'Years'
    title='Carbon concentration increase in lower oceans (ML)'
    ylabel = 'GtC from 1750'
    fig = plotFigure(years, ML, xlabel, ylabel, title)
    pp.savefig(fig)

    DAMAGES = output[13]
    xlabel = 'Years'
    title='Damages (DAMAGES)'
    ylabel = 'trillions 2010 USD per year'
    fig = plotFigure(years, DAMAGES, xlabel, ylabel, title)
    pp.savefig(fig)

    DAMFRAC = output[12]    
    xlabel = 'Years'
    title='Damages as fraction of gross output (DAMFRAC)'
    ylabel = ''
    fig = plotFigure(years, DAMFRAC, xlabel, ylabel, title)
    pp.savefig(fig)

    ABATECOST = output[14]    
    xlabel = 'Years'
    title='Cost of emissions reductions (ABATECOST)'
    ylabel = 'trillions 2010 USD per year'
    fig = plotFigure(years, ABATECOST, xlabel, ylabel, title)
    pp.savefig(fig)
    
    MCABATE = output[15]
    xlabel = 'Years'
    title='Marginal abatement cost(MCABATE)'
    ylabel = '2010 USD per ton CO2'
    fig = plotFigure(years, MCABATE, xlabel, ylabel, title)
    pp.savefig(fig)
        
    E = output[3]
    xlabel = 'Years'
    title='Total CO2 emission (E)'
    ylabel = 'GtCO2 per year'
    fig = plotFigure(years, E, xlabel, ylabel, title)
    pp.savefig(fig)

    MAT = output[6]
    xlabel = 'Years'
    title='Carbon concentration increase in the atmosphere (MAT)'
    ylabel = 'GtC from 1750'
    fig = plotFigure(years, MAT, xlabel, ylabel, title)
    pp.savefig(fig)
    
    FORC = output[9]
    xlabel = 'Years'
    title='Increase in radiative forcing (FORC)'
    ylabel = 'watts per m2 from 1900'
    fig = plotFigure(years, FORC, xlabel, ylabel, title)
    pp.savefig(fig)
    
    RI = output[22]
    xlabel = 'Years'
    title='Real interest rate (RI)'
    ylabel = 'Rate per annum'
    fig = plotFigure(years, RI, xlabel, ylabel, title)
    pp.savefig(fig)

    C = output[20]    
    xlabel = 'Years'
    title='Consumption (C)'
    ylabel = 'trillions 2010 USD per year'
    fig = plotFigure(years, C, xlabel, ylabel, title)
    pp.savefig(fig)
    
    Y = output[18]
    xlabel = 'Years'
    title='Gross product net of abatement and damages (Y)'
    ylabel = 'trillions 2010 USD per year'
    fig = plotFigure(years, Y, xlabel, ylabel, title)
    pp.savefig(fig)
    
    YGROSS = output[1]
    xlabel = 'Years'
    title='World gross product (YGROSS)'
    ylabel = 'trillions 2010 USD per year'
    fig = plotFigure(years, YGROSS, xlabel, ylabel, title)
    pp.savefig(fig)
           
    I = output[19]
    xlabel = 'Years'
    title='Investment (I)'
    ylabel = 'Trillions 2010 USD per year'
    fig = plotFigure(years, I, xlabel, ylabel, title)
    pp.savefig(fig)

    num_times = len(I)

    S = x[num_times:(2*num_times)]
    xlabel = 'Years'
    title='Optimised: Saving rate  (S)'
    ylabel = 'Rate'
    fig = plotFigure(years, S, xlabel, ylabel, title)
    pp.savefig(fig)

    MIU = x[0:num_times]
    title='Optimised: Carbon emission control rate (MIU)'
    xlabel = 'Years'
    ylabel = 'Rate'
    fig = plotFigure(years, MIU, xlabel, ylabel, title)
    pp.savefig(fig)

    pp.close()

###############################################################################
