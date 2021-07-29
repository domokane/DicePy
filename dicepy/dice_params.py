# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:06:52 2021

@author: Dominic
"""

import numpy as np
import csv

class DiceParams():

    """ This class holds the static (over a run) inputs to the DICE model. """

    def __init__(self, num_times, tstep):

        # Maximum cumulative extraction fossil fuels (GtC); denoted by CCum
        self._fosslim = 6000.0 
        # Years per Period
        self._tstep = tstep
        # Indicator where optimized is 1 and base is 0
        self._ifopt = 0

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
            self._cpricebase[i] = self._cprice0 * (1.0 + self._gcprice)**(5.0*(self._t[i]-1))

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
 
###############################################################################

    def runModel(self):
        pass
        
