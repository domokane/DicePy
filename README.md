## DICEPY

This package contains the DICE model as developed by William Nordhaus of Yale University - this is the 2016 version.

It is a globally aggregated model (unlike other regional versions such as the RICE model). 

The model combines neoclassical economic growth theory with the physics of climate change as currently understood to model the evolution of consumption and climate. The aim is to determine an optimal strategy profile for deciding how to manage the challenge of climate change subject to forecasts about a range of climate and economic variables and their inter-relationships.

The steps are

    1. Fossil fuel use generates CO2 emissions
    2. CO2 goes into the atmosphere, ocean and land
    3. The climate system reacts to the CO2 in the atmosphere and its effect on radiative warming, rain, ocean currents and sea-level rises
    4. The modek assesses the resulting impact on agriculture, diseases, ...
    5. This leads to measures to control emissions (taxes, limits, subsidies)
    6. Back to 1

DICE seeks to optimize a path for a number of variables that maximises economic welfare measured as some set of utility functions in partial equilibrium.

DICE represents a social welfare function which increases with population and per capita consumption with a diminishing marginal utility. Different generations differ in their social time preference - the generational discounting and the elasticity of the marginal utility of consumption (consumption elasticity). Together these determine the discount rate.

The model seeks to optimize the flow of consumption over time by setting economic and climate policies. Consumption is general. The aim is to maximise a social welfare function $W$. This is the discounted sum of the population-weighted utility of per capita consumption. $c$ is the per-capita consumption. $l(t)$ is the population as well as labour. $R(t)$ is the discount factor. Hence we have

$W = \sum_{t=1}^T U[c(t), L(t)] R(t) $

The function $U$ captures the utility of consumption. The following function is assumed

$U[c(t),l(t)] = L(t) [c(t)^{1-\alpha}/(1-\alpha)] $

When $\alpha=1$ the utility function is logarithmic. If $\alpha$ is close to zero then utility depends on consumption times labour. But if $\alpha$ is high then increasing consumption does not change utility by as much. This is risk aversion - the more we have the less utility we get from the same increase in consumption. The value of $\alpha$ is linked to the pure rate of time preference. The discount factor is given by

$R(t) = \frac{1}{(1+\rho)^t}$

where $\rho$ is the (flat) rate of social time preference. 