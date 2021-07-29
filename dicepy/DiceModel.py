# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:06:52 2021

@author: Dominic
"""

import numpy as np
import csv
from numba import njit

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
seaborn.set(style='ticks')

###############################################################################

@njit(cache=True, fastmath=True)
def objFn(x, *args):
    """ This is the pass-through function that returns a single float value of
    the objective function for the benefit of the optimisation algorithm. """

    out = simulateDynamics(x, *args)
    return out[0, 0]

###############################################################################


class DiceModel():

    """ This class holds the static (over a run) inputs to the DICE model. """

    def __init__(self, num_times, tstep):

        # Maximum cumulative extraction fossil fuels (GtC); denoted by CCum
        self._fosslim = 6000.0 
        # Years per Period
        self._tstep = tstep
        # Indicator where optimized is 1 and base is 0
        self._ifopt = 0.0

        #######################################################################
        # Economic Preferences
        #######################################################################

        # Elasticity of marginal utility of consumption
        self._elasmu = 1.45 
        # Initial rate of social time preference per year 
        self._prstp = 0.015

        #######################################################################
        # Population and technology
        #######################################################################

        self._gama = 0.300 # Capital elasticity in production function
        self._pop0 = 7403.0 # Initial world population 2015 (millions)
        self._popadj = 0.1340 # Growth rate to calibrate to 2050 pop projection
        self._popasym = 11500.0 # Asymptotic population (millions)
        self._dk = 0.100 # Depreciation rate on capital (per year)
        self._q0 = 105.50 # Initial world gross output 2015 (trill 2010 USD)
        self._k0 = 223.0 # Initial capital value 2015 (trill 2010 USD)
        self._a0 = 5.115 # Initial level of total factor productivity
        self._ga0 = 0.076 # Initial growth rate for TFP per 5 years
        self._dela = 0.0050 # Decline rate of TFP per 5 years

        #######################################################################
        # Emissions parameters
        #######################################################################

        self._gsigma1 = -0.0152  # Initial growth of sigma (per year)
        self._dsig = -0.001  # Decline rate of decarbonization (per period)
        self._eland0 = 2.6  # Carbon emissions from land 2015 (GtCO2 per year)
        self._deland = 0.115 # Decline rate of land emissions (per period)
        self._e0 = 35.85 # Industrial emissions 2015 (GtCO2 per year)
        self._miu0 = 0.03 # Initial emissions control rate for base case 2015
        self._cumetree0 = 100.0 # I ADDED THIS agrees with Nordhaus GAMS code
        self._cca0 = 400.0 # This is an initial condition in Nordhaus GAMS code

        #######################################################################
        # Carbon Cycle Initial Conditions
        #######################################################################

        self._mat0 = 851.0 # Initial Concentration in atmosphere 2015 (GtC)
        self._mu0 = 460.0 # Initial Concentration in upper strata 2015 (GtC)
        self._ml0 = 1740.0 # Initial Concentration in lower strata 2015 (GtC)
        self._mateq = 588.0 # mateq Equilibrium concentration atmosphere  (GtC)
        self._mueq = 360.0 # mueq Equilibrium concentration in upper strata (GtC)
        self._mleq = 1720.0 # mleq Equilibrium concentration in lower strata (GtC)

        #######################################################################
        # Flow parameters, denoted by Phi_ij in the model is a transition matrix
        #######################################################################

        self._b12 = 0.12  # Carbon cycle transition matrix
        self._b23 = 0.0070  # Carbon cycle transition matrix
        self._b11 = 1.0 - self._b12  # Carbon cycle transition matrix
        self._b21 = self._b12 * self._mateq / self._mueq  # Carbon cycle transition matrix
        self._b22 = 1.0 - self._b21 - self._b23  # Carbon cycle transition matrix
        self._b32 = self._b23 * self._mueq / self._mleq  # Carbon cycle transition matrix
        self._b33 = 1.0 - self._b32  # Carbon cycle transition matrix
        self._sig0 = self._e0/(self._q0*(1-self._miu0))  # From Eq. 14

        #######################################################################
        # Climate model parameters
        #######################################################################

        self._t2xco2 = 3.10  # Equilibrium temp impact (oC per doubling CO2)
        self._fex0 = 0.5  # 2015 forcings of non-CO2 GHG (Wm-2)
        self._fex1 = 1.0  # 2100 forcings of non-CO2 GHG (Wm-2)
        self._tocean0 = 0.0068  # Initial lower stratum temp change (C from 1900)
        self._tatm0 = 0.85  # Initial atmospheric temp change (C from 1900)
        self._c1 = 0.1005  # Climate equation coefficient for upper level
        self._c3 = 0.088  # Transfer coefficient upper to lower stratum
        self._c4 = 0.025  # Transfer coefficient for lower level
        self._fco22x = 3.6813  # Forcings of equilibrium CO2 doubling (Wm-2)

        #######################################################################
        # Climate damage parameters
        #######################################################################

        self._a10 = 0.0  # Initial damage intercept
        self._a20 = 0.0  # Initial damage quadratic term
        self._a1 = 0.0  # Damage intercept
        self._a2 = 0.00236  # Damage quadratic term
        self._a3 = 2.00  # Damage exponent

        #######################################################################
        # Abatement cost
        #######################################################################

        self._expcost2 = 2.6  # Theta2, Eq. 10 Exponent of control cost function
        self._pback = 550.0  # Cost of backstop 2010$ per tCO2 2015
        self._gback = 0.025  # Initial cost decline backstop cost per period
        self._limmiu = 1.20  # Upper limit on control rate after 2150
        self._tnopol = 45  # Period before which no emissions controls base
        self._cprice0 = 2.0  # Initial base carbon price (2010$ per tCO2)
        self._gcprice = 0.02  # Growth rate of base carbon price per year

        #######################################################################
        # Scaling and inessential parameters
        # Note that these are unnecessary for the calculations
        # They ensure that MU of first period's consumption =1 and PV cons = PV utilty
        #######################################################################

        self._scale1 = 0.0302455265681763 # Multiplicative scaling coefficient
        self._scale2 = -10993.704 # Additive scaling coefficent

        self._a20 = self._a2
        self._lam = self._fco22x/ self._t2xco2 #From Eq. 25
        self._sig0 = self._e0 / (self._q0 * (1.0 - self._miu0))

        #######################################################################

        self._num_times = num_times
        self._t = np.arange(0, self._num_times+1)

        # Size arrays so that we can index from 1 consistent with matlab
        self._l = np.zeros(num_times+1) # NORDHAUS MAKES THIS L IN HIS EQNS
        self._al = np.zeros(num_times+1) 
        self._sigma = np.zeros(num_times+1)
        self._cumetree = np.zeros(num_times+1)
        self._gsig = np.zeros(num_times+1) 
        self._ga = np.zeros(num_times+1) 
        self._cost1 = np.zeros(num_times+1)
        self._pbacktime = np.zeros(num_times+1)
        self._etree = np.zeros(num_times+1)
        self._rr = np.zeros(num_times+1)
        self._cpricebase = np.zeros(num_times+1)
        
        self._l[1] = self._pop0 # Labor force
        self._ga[1] = self._ga0
        self._al[1] = self._a0
        self._gsig[1] = self._gsigma1
        self._sigma[1]= self._sig0
        self._pbacktime[1] = self._pback
        self._cost1[1] = self._pbacktime[1] * self._sigma[1]  / self._expcost2 / 1000.0
        self._etree[1] = self._eland0
        self._cumetree[1] = self._cumetree0
        self._rr[1] = 1.0 
        self._cpricebase[1] = self._cprice0

        for i in range(2, self._num_times+1):

            self._l[i] = self._l[i-1]*(self._popasym / self._l[i-1])**self._popadj
            self._ga[i] = self._ga0 * np.exp(-self._dela * 5.0 * (self._t[i] - 1.0)) 
            self._al[i] = self._al[i-1]/(1.0 - self._ga[i-1])
            self._gsig[i] = self._gsig[i-1]*((1.0 + self._dsig)**self._tstep)
            self._sigma[i] = self._sigma[i-1] * np.exp(self._gsig[i-1] * self._tstep)
            self._pbacktime[i] = self._pback * (1.0 - self._gback) ** (self._t[i]-1) 
            self._cost1[i] = self._pbacktime[i] * self._sigma[i]  / self._expcost2 / 1000.0
            self._etree[i] = self._eland0 * (1.0 - self._deland) ** (self._t[i]-1) 
            self._cumetree[i] = self._cumetree[i-1] + self._etree[i-1]*(5.0/3.666)
            self._rr[i] = 1.0 / ((1.0 + self._prstp)**(self._tstep * (self._t[i]-1))) 
            self._cpricebase[i] = self._cprice0 * (1.0 + self._gcprice)**(5*(self._t[i]-1))

        #The following three equations define the exogenous radiative forcing; 
        self._forcoth = np.zeros(self._num_times+1)
        self._forcoth[1] = self._fex0

        for i in range(2, 18):
            self._forcoth[i] = self._fex0 + (1.0/17.0) * (self._fex1 - self._fex0) * (self._t[i]-1)

        for i in range(18, self._num_times+1):
            self._forcoth[i] = self._fex1

        #Optimal long-run savings rate used for transversality (Question)        
        self._optlrsav = (self._dk + .004)/(self._dk + .004 * self._elasmu + self._prstp) * self._gama 

        if 1==1:

            f = open("./results/parameters.csv" , mode = "w", newline='')
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
            header = []
            header.append("PERIOD")
            header.append("L")
            header.append("GA")
            header.append("AL")
            header.append("GSIG")
            header.append("SIGMA")
            header.append("PBACKTIME")
            header.append("COST1")
            header.append("ETREE")
            header.append("CUMETREE")
            header.append("RR")
            header.append("CPRICEBASE")
            writer.writerow(header)
            
            num_rows = self._num_times + 1
        
            for i in range(0, num_rows):
                row = []
                row.append(i)
                row.append(self._l[i])
                row.append(self._ga[i])
                row.append(self._al[i])
                row.append(self._gsig[i])
                row.append(self._sigma[i])
                row.append(self._pbacktime[i])
                row.append(self._cost1[i])
                row.append(self._etree[i])
                row.append(self._cumetree[i])
                row.append(self._rr[i])
                row.append(self._cpricebase[i])

                writer.writerow(row)

            f.close()

        elif 1==2:

            print("Labour:", self._l) # CHECKED OK
            print("Growth rate of productivity", self._ga) # CHECKED OK
            print("Productivity", self._al) # CHECKED OK
            print("CO2 output ratio:", self._sigma) # CHECKED OK
            print("Change in sigma", self._gsig) # CHECKED OK
            print("Cumulative from land", self._cumetree) # CHECKED BUT UNSURE ABOUT INITIAL PRICE SOURCE OF 100
            print("Adjusted cost backstop", self._cost1) # CHECKED OK
            print("Backstop price", self._pbacktime) # CHECKED OK
            print("Emissions deforestation", self._etree) # CHECKED OK
            print("Utility social rate discount factor", self._rr) # CANNOT SEE ON NORDHAUS ??
            print("Carbon price based case", self._cpricebase) # AGREES WITH NORDHAUS UNTIL 2235 ??
            print("Exogenous forcing others", self._forcoth) # CHECKED OK
            print("Long run savings rate", self._optlrsav) # CHECKED OK

        else:
            print("SOME CHECKING TO BE DONE")
 
#    @njit(cache=True, fastmath=True)
    def simulateDynamics(self, x, sign, outputType, num_times,
                         tstep, al, ll, sigma, cumetree, forcoth,
                         cost1, etree,
                         scale1, scale2,
                         ml0, mu0, mat0, cca0,
                         a1, a2, a3,
                         c1, c3, c4,
                         b11, b12, b21, b22, b32, b23, b33,
                         fco22x, t2xco2, rr, gama,
                         tocean0, tatm0, elasmu, prstp, expcost2,
                         k0, dk, pbacktime):
        """ This is the simulation of the DICE 2016 model dynamics. It is optimised
        for speed. For this reason I have avoided the use of classes. """
    
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
        MAT[1] = mat0
        ML[1] = ml0
        MU[1] = mu0
        TATM[1] = tatm0
        TOCEAN[1] = tocean0
        K[1] = k0
    
        YGROSS[1] = al[1] * ((L[1]/MILLE)**(1.0-gama)) * K[1]**gama
        EIND[1] = sigma[1] * YGROSS[1] * (1.0 - MIUopt[1])
        E[1] = EIND[1] + etree[1]
        CCA[1] = cca0  # DOES NOT START TILL PERIOD 2
        CCATOT[1] = CCA[1] + cumetree[1]
        FORC[1] = fco22x * np.log(MAT[1]/588.000)/LOG2 + forcoth[1]
        DAMFRAC[1] = a1*TATM[1] + a2*TATM[1]**a3
        DAMAGES[1] = YGROSS[1] * DAMFRAC[1]
        ABATECOST[1] = YGROSS[1] * cost1[1] * MIUopt[1]**expcost2
        MCABATE[1] = pbacktime[1] * MIUopt[1]**(expcost2-1)
        CPRICE[1] = pbacktime[1] * (MIUopt[1])**(expcost2-1)
    
        YNET[1] = YGROSS[1] * (1.0 - DAMFRAC[1])
        Y[1] = YNET[1] - ABATECOST[1]
        II[1] = Sopt[1] * Y[1]
        C[1] = Y[1] - II[1]
        CPC[1] = MILLE * C[1] / L[1]
        # RI[T] is set at end
    
        PERIODU[1] = ((C[1]*MILLE/L[1])**(1.0-elasmu)-1.0) / (1.0 - elasmu) - 1.0
        CEMUTOTPER[1] = PERIODU[1] * L[1] * rr[1]
    
        for i in range(2, num_times+1):
    
            # Depend on t-1
            CCA[i] = CCA[i-1] + EIND[i-1] * 5.0 / 3.666
            MAT[i] = MAT[i-1] * b11 + MU[i-1] * b21 + E[i-1] * 5.0 / 3.666
            ML[i] = ML[i-1] * b33 + MU[i-1] * b23
            MU[i] = MAT[i-1] * b12 + MU[i-1] * b22 + ML[i-1]*b32
            TOCEAN[i] = TOCEAN[i-1] + c4 * (TATM[i-1] - TOCEAN[i-1])
    
            CCATOT[i] = CCA[i] + cumetree[i]
    
            K[i] = (1.0-dk)**tstep * K[i-1] + tstep * II[i-1]
            YGROSS[i] = al[i] * ((L[i]/MILLE)**(1.0 - gama)) * K[i]**gama
            EIND[i] = sigma[i] * YGROSS[i] * (1.0 - MIUopt[i])
            E[i] = EIND[i] + etree[i]
    
            FORC[i] = fco22x * np.log(MAT[i]/588.000)/LOG2 + forcoth[i]
            TATM[i] = TATM[i-1] + c1 * \
                (FORC[i] - (fco22x/t2xco2) * TATM[i-1] -
                 c3 * (TATM[i-1] - TOCEAN[i-1]))
    
            DAMFRAC[i] = a1 * TATM[i] + a2*TATM[i]**a3
            DAMAGES[i] = YGROSS[i] * DAMFRAC[i]
    
            ABATECOST[i] = YGROSS[i] * cost1[i] * MIUopt[i]**expcost2
            MCABATE[i] = pbacktime[i] * MIUopt[i]**(expcost2-1)
            CPRICE[i] = pbacktime[i] * (MIUopt[i])**(expcost2-1)
    
            YNET[i] = YGROSS[i] * (1.0 - DAMFRAC[i])
            Y[i] = YNET[i] - ABATECOST[i]
    
            II[i] = Sopt[i] * Y[i]
            C[i] = Y[i] - II[i]
            CPC[i] = MILLE * C[i] / L[i]
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

###############################################################################

    def runModel(self):
        pass

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
                     k0, dk, pbacktime):
    """ This is the simulation of the DICE 2016 model dynamics. It is optimised
    for speed. For this reason I have avoided the use of classes. """

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
    MAT[1] = mat0
    ML[1] = ml0
    MU[1] = mu0
    TATM[1] = tatm0
    TOCEAN[1] = tocean0
    K[1] = k0

    YGROSS[1] = al[1] * ((L[1]/MILLE)**(1.0-gama)) * K[1]**gama
    EIND[1] = sigma[1] * YGROSS[1] * (1.0 - MIUopt[1])
    E[1] = EIND[1] + etree[1]
    CCA[1] = cca0  # DOES NOT START TILL PERIOD 2
    CCATOT[1] = CCA[1] + cumetree[1]
    FORC[1] = fco22x * np.log(MAT[1]/588.000)/LOG2 + forcoth[1]
    DAMFRAC[1] = a1*TATM[1] + a2*TATM[1]**a3
    DAMAGES[1] = YGROSS[1] * DAMFRAC[1]
    ABATECOST[1] = YGROSS[1] * cost1[1] * MIUopt[1]**expcost2
    MCABATE[1] = pbacktime[1] * MIUopt[1]**(expcost2-1)
    CPRICE[1] = pbacktime[1] * (MIUopt[1])**(expcost2-1)

    YNET[1] = YGROSS[1] * (1.0 - DAMFRAC[1])
    Y[1] = YNET[1] - ABATECOST[1]
    II[1] = Sopt[1] * Y[1]
    C[1] = Y[1] - II[1]
    CPC[1] = MILLE * C[1] / L[1]
    # RI[T] is set at end

    PERIODU[1] = ((C[1]*MILLE/L[1])**(1.0-elasmu)-1.0) / (1.0 - elasmu) - 1.0
    CEMUTOTPER[1] = PERIODU[1] * L[1] * rr[1]

    for i in range(2, num_times+1):

        # Depend on t-1
        CCA[i] = CCA[i-1] + EIND[i-1] * 5.0 / 3.666
        MAT[i] = MAT[i-1] * b11 + MU[i-1] * b21 + E[i-1] * 5.0 / 3.666
        ML[i] = ML[i-1] * b33 + MU[i-1] * b23
        MU[i] = MAT[i-1] * b12 + MU[i-1] * b22 + ML[i-1]*b32
        TOCEAN[i] = TOCEAN[i-1] + c4 * (TATM[i-1] - TOCEAN[i-1])

        CCATOT[i] = CCA[i] + cumetree[i]

        K[i] = (1.0-dk)**tstep * K[i-1] + tstep * II[i-1]
        YGROSS[i] = al[i] * ((L[i]/MILLE)**(1.0 - gama)) * K[i]**gama
        EIND[i] = sigma[i] * YGROSS[i] * (1.0 - MIUopt[i])
        E[i] = EIND[i] + etree[i]

        FORC[i] = fco22x * np.log(MAT[i]/588.000)/LOG2 + forcoth[i]
        TATM[i] = TATM[i-1] + c1 * \
            (FORC[i] - (fco22x/t2xco2) * TATM[i-1] -
             c3 * (TATM[i-1] - TOCEAN[i-1]))

        DAMFRAC[i] = a1 * TATM[i] + a2*TATM[i]**a3
        DAMAGES[i] = YGROSS[i] * DAMFRAC[i]

        ABATECOST[i] = YGROSS[i] * cost1[i] * MIUopt[i]**expcost2
        MCABATE[i] = pbacktime[i] * MIUopt[i]**(expcost2-1)
        CPRICE[i] = pbacktime[i] * (MIUopt[i])**(expcost2-1)

        YNET[i] = YGROSS[i] * (1.0 - DAMFRAC[i])
        Y[i] = YNET[i] - ABATECOST[i]

        II[i] = Sopt[i] * Y[i]
        C[i] = Y[i] - II[i]
        CPC[i] = MILLE * C[i] / L[i]
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
