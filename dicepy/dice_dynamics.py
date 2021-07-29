# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 23:20:04 2021

@author: Dominic
"""

import csv
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
seaborn.set(style='ticks')


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
    return out[0, 0]

###############################################################################

@njit(cache=True, fastmath=True)
def simulateDynamics(x, sign, outputType, num_times,
                     tstep, al, ll, sigma, cumetree, forcoth,
                     cost1, etree,
                     scale1, scale2,
                     ml0, mu0, mat0, cca0,
                     a1, a2, a3,
                     c1, c3, c4,
                     b11, b12, b21, b22, b32, b23, b33,
                     fco22x, t2xco2, rr, gama,
                     tocean0, tatm0, elasmu, prstp, expcost2,
                     k0, dk, pbacktime, 
                     scc_period, e_bump, c_bump):
    """ This is the simulation of the DICE 2016 model dynamics. It is optimised
    for speed. For this reason I have avoided the use of classes. """

#    print(scc_period, e_bump, c_bump)
    
    LOG2 = np.log(2)
    L = ll  # NORDHAUS RENAMES IT TO UPPER CASE IN EQUATIONS
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
    II = np.zeros(num_times+1)
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
    CCA[1] = cca0
    K[1] = k0
    MAT[1] = mat0
    MU[1] = mu0
    ML[1] = ml0
    TATM[1] = tatm0
    TOCEAN[1] = tocean0

    YGROSS[1] = al[1] * ((L[1]/MILLE)**(1.0-gama)) * K[1]**gama
    EIND[1] = sigma[1] * YGROSS[1] * (1.0 - MIUopt[1])
    E[1] = EIND[1] + etree[1]

    if scc_period == 1:
        E[1] = E[1] + e_bump

    CCATOT[1] = CCA[1] + cumetree[1]
    FORC[1] = fco22x * np.log(MAT[1]/588.0)/LOG2 + forcoth[1]
    DAMFRAC[1] = a1*TATM[1] + a2*TATM[1]**a3
    DAMAGES[1] = YGROSS[1] * DAMFRAC[1]
    ABATECOST[1] = YGROSS[1] * cost1[1] * MIUopt[1]**expcost2
    MCABATE[1] = pbacktime[1] * MIUopt[1]**(expcost2-1)
    CPRICE[1] = pbacktime[1] * (MIUopt[1])**(expcost2-1)

    YNET[1] = YGROSS[1] * (1.0 - DAMFRAC[1])
    Y[1] = YNET[1] - ABATECOST[1]
    II[1] = Sopt[1] * Y[1]
    C[1] = Y[1] - II[1]

    if scc_period == 1:
        C[1] = C[1] + c_bump

    CPC[1] = MILLE * C[1] / L[1]
    # RI[T] is set at end

    PERIODU[1] = ((C[1]*MILLE/L[1])**(1.0-elasmu)-1.0) / (1.0 - elasmu) - 1.0
    CEMUTOTPER[1] = PERIODU[1] * L[1] * rr[1]

    # Reference 
    # http://www.econ.yale.edu/~nordhaus/homepage/homepage/DICE2016R-091916ap.gms

    eta = (fco22x/t2xco2)

    for i in range(2, num_times+1):
            
        # Depend on t-1
        CCA[i] = CCA[i-1] + EIND[i-1] * (5.0 / 3.666)

        MAT[i] = max(10.0, MAT[i-1] * b11 + MU[i-1] * b21 + E[i-1] * (5.0 / 3.666))
        MU[i] = max(100.0, MAT[i-1] * b12 + MU[i-1] * b22 + ML[i-1] * b32)
        ML[i] = max(1000.0, ML[i-1] * b33 + MU[i-1] * b23)

        TOCEAN[i] = max(-1.0, TOCEAN[i-1] + c4 * (TATM[i-1] - TOCEAN[i-1]))
        TOCEAN[i] = min(20.0, TOCEAN[i])
        
        CCATOT[i] = CCA[i] + cumetree[i]

        # Depend on t
        K[i] = max(1.0, (1.0-dk)**tstep * K[i-1] + tstep * II[i-1])
        YGROSS[i] = al[i] * ((L[i]/MILLE)**(1.0 - gama)) * K[i]**gama
        EIND[i] = sigma[i] * YGROSS[i] * (1.0 - MIUopt[i])
        E[i] = EIND[i] + etree[i]

        if scc_period == i:
            E[i] = E[i] + e_bump

        FORC[i] = fco22x * np.log(MAT[i]/588.000)/LOG2 + forcoth[i]
        TATM[i] = TATM[i-1] + c1 * (FORC[i] - eta * TATM[i-1] - c3 * (TATM[i-1] - TOCEAN[i-1]))
        TATM[i] = max(-1.0, TATM[i])
        TATM[i] = min(12.0, TATM[i]) # Nordhaus has this twice at 12 and 20 so I use the 20

        DAMFRAC[i] = a1 * TATM[i] + a2*(TATM[i]**a3)
        DAMAGES[i] = YGROSS[i] * DAMFRAC[i]

        ABATECOST[i] = YGROSS[i] * cost1[i] * MIUopt[i]**expcost2
        MCABATE[i] = pbacktime[i] * MIUopt[i]**(expcost2-1)
        CPRICE[i] = pbacktime[i] * (MIUopt[i])**(expcost2-1)

        YNET[i] = YGROSS[i] * (1.0 - DAMFRAC[i])
        Y[i] = YNET[i] - ABATECOST[i]

        II[i] = Sopt[i] * Y[i]
        C[i] = max(2.0, Y[i] - II[i])
 
        if scc_period == i:
            C[i] = C[i] + c_bump

        CPC[i] = max(0.01, MILLE * C[i] / L[i])

        PERIODU[i] = ((C[i]*MILLE/L[i])**(1.0-elasmu) - 1.0) / \
            (1.0 - elasmu) - 1.0

        CEMUTOTPER[i] = PERIODU[i] * L[i] * rr[i]

    for i in range(1, num_times):
        RI[i] = (1.0 + prstp) * (CPC[i+1]/CPC[i])**(elasmu/tstep) - 1.0

    RI[-1] = 0.0
    
    output = np.zeros((num_times, 50))

    if outputType == 0:

        resUtility = tstep * scale1 * np.sum(CEMUTOTPER) + scale2
        resUtility *= sign
        output[0, 0] = resUtility
#        print(output[0,0])
        return output

    elif outputType == 1:

        # EXTRA VALUES COMPUTED LATER
        CO2PPM = np.zeros(num_times+1)
        for i in range(1, num_times):
            CO2PPM[i] = MAT[i] / 2.13

        SOCCC = np.zeros(num_times+1)
        for i in range(1, num_times):
            SOCCC[i] = -999.0

        for iTime in range(1, num_times+1):

            col = 0
            jTime = iTime - 1
            output[jTime, col] = EIND[iTime]
            col += 1  # 0
            output[jTime, col] = E[iTime]
            col += 1  # 1
            output[jTime, col] = CO2PPM[iTime]
            col += 1  # 2
            output[jTime, col] = TATM[iTime]
            col += 1  # 3
            output[jTime, col] = Y[iTime]
            col += 1  # 4
            output[jTime, col] = DAMFRAC[iTime]
            col += 1  # 5
            output[jTime, col] = CPC[iTime]
            col += 1  # 6
            output[jTime, col] = CPRICE[iTime]
            col += 1  # 7
            output[jTime, col] = MIUopt[iTime]
            col += 1  # 8
            output[jTime, col] = RI[iTime]
            col += 1  # 9
            output[jTime, col] = SOCCC[iTime]
            col += 1  # 10

            output[jTime, col] = ll[iTime]
            col += 1  # 11
            output[jTime, col] = al[iTime]
            col += 1  # 12
            output[jTime, col] = YGROSS[iTime]
            col += 1  # 13

            output[jTime, col] = K[iTime]
            col += 1  # 14
            output[jTime, col] = Sopt[iTime]
            col += 1  # 15
            output[jTime, col] = II[iTime]
            col += 1  # 16
            output[jTime, col] = YNET[iTime]
            col += 1  # 17

            output[jTime, col] = CCA[iTime]
            col += 1  # 18
            output[jTime, col] = CCATOT[iTime]
            col += 1  # 19
            output[jTime, col] = ML[iTime]
            col += 1  # 20
            output[jTime, col] = MU[iTime]
            col += 1  # 21
            output[jTime, col] = FORC[iTime]
            col += 1  # 22
            output[jTime, col] = TOCEAN[iTime]
            col += 1  # 23
            output[jTime, col] = DAMAGES[iTime]
            col += 1  # 24
            output[jTime, col] = ABATECOST[iTime]
            col += 1  # 25
            output[jTime, col] = MCABATE[iTime]
            col += 1  # 26
            output[jTime, col] = C[iTime]
            col += 1  # 27
            output[jTime, col] = PERIODU[iTime]
            col += 1  # 28
            output[jTime, col] = CEMUTOTPER[iTime]
            col += 1  # 29
            output[jTime, col] = MAT[iTime]
            col += 1  # 30

        return output

    else:
        raise Exception("Unknown output type.")

    return output

###############################################################################


def dumpState(years, output, filename):

    f = open(filename, mode="w", newline='')
    writer = csv.writer(f, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)

    header = []
    header.append("EIND")
    header.append("E")
    header.append("CO2PPM")
    header.append("TATM")
    header.append("Y")
    header.append("DAMFRAC")
    header.append("CPC")
    header.append("CPRICE")
    header.append("MIUopt")
    header.append("RI")
    header.append("SOCCC")

    header.append("L")
    header.append("AL")
    header.append("YGROSS")

    header.append("K")
    header.append("Sopt")
    header.append("I")
    header.append("YNET")

    header.append("CCA")
    header.append("CCATOT")
    header.append("ML")
    header.append("MU")
    header.append("FORC")
    header.append("TOCEAN")
    header.append("DAMAGES")
    header.append("ABATECOST")
    header.append("MCABATE")
    header.append("C")
    header.append("PERIODU")
    header.append("CEMUTOTPER")
    header.append("MAT")

    if 1 == 0:
        num_cols = output.shape[0]
        num_rows = len(header)

        row = ["INDEX"]
        for iCol in range(0, num_cols):
            row.append(iCol+1)
        writer.writerow(row)

        for iRow in range(1, num_rows):
            row = [header[iRow-1]]
            for iCol in range(0, num_cols):
                row.append(output[iCol, iRow-1])
            writer.writerow(row)
    else:
        num_rows = output.shape[0]
        num_cols = len(header)

        row = ['IPERIOD']
        for iCol in range(0, num_cols):
            row.append(header[iCol])
        writer.writerow(row)

        for iRow in range(1, num_rows):
            row = [iRow]
            for iCol in range(1, num_cols):
                row.append(output[iRow, iCol-1])
            writer.writerow(row)

    f.close()

###############################################################################


def plotFigure(x, y, xlabel, ylabel, title):

    x = x[:-1]
    y = y[:-1]

#    mpl.style.use('ggplot')
    fig = plt.figure(figsize=(8, 6), dpi=72, facecolor="white")
#    axes = plt.subplot(111)
    plt.plot(x, y)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=14)
    seaborn.despine()
    fig.tight_layout()
    return fig

###############################################################################


def plotStateToFile(fileName, years, output, x):

    num_times = output.shape[0]
    output = np.transpose(output)

    pp = PdfPages(fileName)

    TATM = output[3]
    title = 'Change in Atmosphere Temperature (TATM)'
    xlabel = 'Years'
    ylabel = 'Degrees C from 1900'
    fig = plotFigure(years, TATM, xlabel, ylabel, title)
    pp.savefig(fig)

    TOCEAN = output[23]
    xlabel = 'Years'
    title = 'Change in Ocean Temperature (TOCEAN)'
    ylabel = 'Degrees C from 1900'
    fig = plotFigure(years, TOCEAN, xlabel, ylabel, title)
    pp.savefig(fig)

    MU = output[21]
    xlabel = 'Years'
    title = 'Change in Carbon Concentration Increase in Upper Oceans (MU)'
    ylabel = 'GtC from 1750'
    fig = plotFigure(years, MU, xlabel, ylabel, title)
    pp.savefig(fig)

    ML = output[20]
    xlabel = 'Years'
    title = 'Change in Carbon Concentration in Lower Oceans (ML)'
    ylabel = 'GtC from 1750'
    fig = plotFigure(years, ML, xlabel, ylabel, title)
    pp.savefig(fig)

    DAMAGES = output[24]
    xlabel = 'Years'
    title = 'Damages (DAMAGES)'
    ylabel = 'USD (2010) Trillions per Year'
    fig = plotFigure(years, DAMAGES, xlabel, ylabel, title)
    pp.savefig(fig)

    DAMFRAC = output[5]
    xlabel = 'Years'
    title = 'Damages as a Fraction of Gross Output (DAMFRAC)'
    ylabel = 'Damages Output Ratio'
    fig = plotFigure(years, DAMFRAC, xlabel, ylabel, title)
    pp.savefig(fig)

    ABATECOST = output[25]
    xlabel = 'Years'
    title = 'Cost of Emissions Reductions (ABATECOST)'
    ylabel = 'USD (2010) Trillions per Year'
    fig = plotFigure(years, ABATECOST, xlabel, ylabel, title)
    pp.savefig(fig)

    MCABATE = output[26]
    xlabel = 'Years'
    title = 'Marginal abatement cost(MCABATE)'
    ylabel = '2010 USD per Ton of CO2'
    fig = plotFigure(years, MCABATE, xlabel, ylabel, title)
    pp.savefig(fig)

    E = output[1]
    xlabel = 'Years'
    title = 'Total CO2 emission (E)'
    ylabel = 'GtCO2 per year'
    fig = plotFigure(years, E, xlabel, ylabel, title)
    pp.savefig(fig)

    EIND = output[0]
    xlabel = 'Years'
    title = 'Total Industrial CO2 emissions (EIND)'
    ylabel = 'GtCO2 per year'
    fig = plotFigure(years, EIND, xlabel, ylabel, title)
    pp.savefig(fig)

    MAT = output[30]
    xlabel = 'Years'
    title = 'Change in Carbon Concentration in Atmosphere (MAT)'
    ylabel = 'GtC from 1750'
    fig = plotFigure(years, MAT, xlabel, ylabel, title)
    pp.savefig(fig)

    FORC = output[22]
    xlabel = 'Years'
    title = 'Increase in Radiative Forcing (FORC)'
    ylabel = 'Watts per M2 from 1900'
    fig = plotFigure(years, FORC, xlabel, ylabel, title)
    pp.savefig(fig)

    RI = output[9]
    xlabel = 'Years'
    title = 'Real Interest Rate (RI)'
    ylabel = 'Rate per annum'
    fig = plotFigure(years, RI, xlabel, ylabel, title)
    pp.savefig(fig)

    C = output[27]
    xlabel = 'Years'
    title = 'Consumption (C)'
    ylabel = 'USD (2010) Trillion per Year'
    fig = plotFigure(years, C, xlabel, ylabel, title)
    pp.savefig(fig)

    Y = output[4]
    xlabel = 'Years'
    title = 'Gross Product Net of Abatement and Damages (Y)'
    ylabel = 'USD (2010) Trillion per Year'
    fig = plotFigure(years, Y, xlabel, ylabel, title)
    pp.savefig(fig)

    YGROSS = output[13]
    xlabel = 'Years'
    title = 'World Gross Product (YGROSS)'
    ylabel = 'USD (2010) Trillion per Year'
    fig = plotFigure(years, YGROSS, xlabel, ylabel, title)
    pp.savefig(fig)

    II = output[16]
    xlabel = 'Years'
    title = 'Investment (I)'
    ylabel = 'USD (2010) Trillion per Year'
    fig = plotFigure(years, II, xlabel, ylabel, title)
    pp.savefig(fig)

    num_times = len(II)

    S = x[num_times:(2*num_times)]
    xlabel = 'Years'
    title = 'Optimised: Saving Rates (S)'
    ylabel = 'Rate'
    fig = plotFigure(years, S, xlabel, ylabel, title)
    pp.savefig(fig)

    MIU = x[0:num_times]
    title = 'Optimised: Carbon Emission Control Rate (MIU)'
    xlabel = 'Years'
    ylabel = 'Rate'
    fig = plotFigure(years, MIU, xlabel, ylabel, title)
    pp.savefig(fig)

    pp.close()

###############################################################################
