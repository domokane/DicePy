# DICEPY
## A Python implementation of Nordhaus' 2016 DICE Model

This package contains the DICE model as developed by William Nordhaus of Yale University - this is the 2016 version.

It is a globally aggregated model (unlike other regional versions such as the RICE model). 

The following is a summary of the model taken from Nordhaus (2016).

The model combines neoclassical economic growth theory with the physics of climate change as currently understood to model the evolution of consumption and climate. The aim is to determine an optimal strategy profile for deciding how to manage the challenge of climate change subject to forecasts about a range of climate and economic variables and their inter-relationships.

The steps are

    1. Fossil fuel use generates CO2 emissions
    2. CO2 goes into the atmosphere, ocean and land
    3. The effect of this on radiative warming, rain, ocean currents and sea-level rises
    4. The modek assesses the resulting impact on agriculture, diseases, ...
    5. This leads to measures to control emissions (taxes, limits, subsidies)
    6. Back to 1

We now consider the economic and geophysical components of the model separately.

## Economic Factors

DICE seeks to optimize a path for a number of variables that maximises economic welfare measured as some set of utility functions in partial equilibrium.

DICE represents a social welfare function which increases with population and per capita consumption with a diminishing marginal utility. Different generations differ in their social time preference - the generational discounting and the elasticity of the marginal utility of consumption (consumption elasticity). Together these determine the discount rate.

The model seeks to optimize the flow of consumption over time by setting economic and climate policies. Consumption is general. The aim is to maximise a social welfare function $W$. This is the discounted sum of the population-weighted utility of per capita consumption. $c$ is the per-capita consumption. $l(t)$ is the population as well as labour. $R(t)$ is the discount factor. Hence we have
$$W = \sum_{t=1}^T U[c(t), L(t)] R(t)$$.
The function $U$ captures the utility of consumption. The following function is assumed
$$U[c(t),l(t)] = L(t) [c(t)^{1-\alpha}/(1-\alpha)]$$

When $\alpha=1$ the utility function is logarithmic. If $\alpha$ is close to zero then utility depends on consumption times labour. But if $\alpha$ is high then increasing consumption does not change utility by as much. This is risk aversion - the more we have the less utility we get from the same increase in consumption. The value of $\alpha$ is linked to the pure rate of time preference. The discount factor is given by
$$R(t) = \frac{1}{(1+\rho)^t}$$
where $\rho$ is the (flat) rate of social time preference.

The DICE model assumes a single commodity that is used for consumption, investment or abatement. Output, population and emissions are built up from national data and aggregated using PPP exchange rates. Population and labour force are exogenous. They are logistic of the form
$$L(t) = L(t-1) (1+g_L(t))$$
where 
$$g_L(t) = g_L(t-1)/(1+\delta_L))$$. 
The growth rate is set to decline so that the global population reaches a limit of 10.5 billion in 2100.

Output is produced using a Cobb-Douglas production function in capital, labour and energy. Technological change is split into two forms. FIrst is economy-wide technological change and second is carbobn-saving technoloical change. The total factor productivity TFP represented By $A(t)$ is a logistic function 
$$A(t) = A(t-1) (1+g_A(t))$$
where 
$$g_A(t) = g_A(t-1)/(1+\delta_A)$$. 
TFP growth declines over time. In 2015 $g_A(2015)=7.9\%$ per 5-year period and $\delta_A=0.6\%$ per five years. The growth in consumption per capital is $1.9\%$ per year from 2010 to 2100 and $0.9\%$ from 2100 to 2200.

Outputs are measured in PPP units and production functions are assumed to be Cobb-Douglas constant returns to scale in capital, labour and Hicks-neutral technological change. Gloabl output follows
$$Q(t) = [1 - \Gamma(t)] A(t) K(t)^{\gamma} L(t)^{1-\gamma} / [1+\Omega(t))]$$
This is output net of damages and abatement costs. $A(t)$ is total factor productivity. $K(t)$ is capital stock and services. Variables $\Omega(t)$ and $\Gamma(t)$ represent climate damages and abatement costs. Damages are assumed to be a quadratic function of temperature change and is calibrated for temperature changes in the range 0-3C. The damages function ensures that damages do not exceed 100% of output. This limits the usefulness of this model for catastrophic climate change. The damages is given by
$$\Omega(t) = \psi_1 T_{AT} (t) + \psi_2 T_{AT}^{2}$$.
There is a typo in this equation in the Nordhaus user guide. The abatement cost function is given by
$$\Gamma(t) = \theta_1 (t) \mu(t)^{\theta_2}$$
where $\mu(t)$ is the emissions reduction rate. The abatement function assumes abatement costs are proportional to output and to a power function of the reduction rate $\mu(t)$. The marginal costs of reductions rises from zero more than linearly in $\mu(t)$.

There are three standard accounting equations:
- Output is consumption plus investment, i.e. $Q(t) = C(t) + I(t)$
- Per capital consumption $c(t) = C(t) / L(t)$
- Capital stocks follows a perpetual inventory method with an exponential depreciation rate $K(t) = I(t) - \delta_K K(t-1)$

CO2 emissions are projected as as function of total output, a time-varying emissions output ratio and the emissions control rate. 

The carbon price is determined by setting it equal to the marginal cost of emissions. This is calculated from the abatememt cost equation for $\Gamma(t)$ after substituing the output equations.

The final two equations are the emissions equation and the resource constraint on carbon fuels. The emissions $E(t)$ are 

$$E_{Ind}(t) = \sigma(t) [1-\mu(t)] A(t) K(t)^{\gamma} L(t)^{1-\gamma}$$

The carbon intensity $\sigma(t)$ is exogenous. They are reduced in each period by the emissions-reductions rate $(1-\mu(t))$. The carbon intensity is built from emissions estimates. We have
$$\sigma(t) = \sigma(t-1) [1+g_{\sigma}(t)]$$
where
$$g_{\sigma} (t) = g_{\sigma}(t-1) / (1+ \delta_{\sigma})$$.
The carbon intensity is initialised to the carbon intensity in 20XX. It is in units of tons per \$1000 of GDP. The growth rate is -1.0% per year. And $\delta_{\sigma} = -0.10\%$. The following equation
$$CCum \ge \sum_{t=1}^T E_{Ind}(t)$$
is the limitation on total resources of carbon fuels. In Dice 2013 it is set to 6000 tons of carbon content.

## Geophysical Factors

The model links economic activity and greenhouse-gas emissions to the carbon cycle, radiative forcings and climate change. The only GHG modelled is CO2. We saw earlier the production of industrial emissions $E_{Ind}(t)$. Other CO2 comes from land-use changes, other well-mixed GHGs and aerosols. Total emissions is given by
$$E(t) = E_{Ind}(t) + E_{Land}(t)$$.
The carbon cycle is modelled as three components - atmosphere (AT), upper oceans (UP) and deep oceans (LO). These act as carbon dioxide reservoirs which mix, albeit at varying speeds. The dynamical equations are

$$M_{AT}(t) = E(t) + \phi_{11} M_{AT} (t-1) + \phi_{21} M_{UP} (t-1)$$
$$M_{UP}(t) = \phi_{12} M_{AT} (t-1) + \phi_{22} M_{UP} (t-1) + \phi_{32} M_{LO} (t-1)$$
$$M_{LO}(t) = \phi_{23} M_{UP} (t-1) + \phi_{33} M_{LO} (t-1)$$

The $\phi$ represent flows between these different reservoirs for CO2. We know that 35% of CO2 gases emitted stay in the atmosphere for over 100 years. The next step concerns the link between GHG amd climate change. We have one equation for the radiative forcing and two for the climate system.

The radiative forcing is given by
$$F(t) = F_{EX} + \eta \ln_2 \left[ M_{AT}(t)/M_{AT}(2750) \right] $$
to switch to natural logs we have
$$F(t) = F_{EX} + \eta \ln \left[ M_{AT}(t)/M_{AT}(2750) \right] / \ln 2$$

The model also captures lags in warming.
$$T_{AT}(t) = \phi_{11} M_{AT} (t-1) + \phi_{21} M_{UP} (t-1)$$
$$T_{LO}(t) = \phi_{12} M_{AT} (t-1) + \phi_{22} M_{UP} (t-1) + \phi_{32} M_{LO} (t-1)$$

