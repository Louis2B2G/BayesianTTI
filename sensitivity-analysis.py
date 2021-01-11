import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tti import simulation
from emukit.test_functions.sensitivity import Ishigami
from emukit.core import ContinuousParameter, ParameterSpace

np.random.seed(10) # for reproducibility

# Set the domain over which to perform the sensitiviy analysis
probability_domain = (0,1)

space = ParameterSpace(
          [ContinuousParameter('p_under18', *probability_domain), 
           ContinuousParameter('compliance', *probability_domain)])

config_details = {0 : {"name": "p_under18", "config": "case_config"}, 
                  1 : {"name": "compliance", "config": "policy_config"}}

# Run the simulation a number of times to get some datapoints for the emulator
from emukit.core.initial_designs import RandomDesign

design = RandomDesign(space)
x = design.get_samples(100)
# NB this takes a while to run
y = np.array([simulation(k, config_details)['Effective R'] for k in x])[:,np.newaxis]

# Use GP regression as the emulator
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity

model_gpy = GPRegression(x,y)
model_emukit = GPyModelWrapper(model_gpy)
model_emukit.optimize()

# Run Monte Carlo estimation of Sobol indices on the emulator
num_monte_carlo = 10000
sens_gpbased = MonteCarloSensitivity(model = model_emukit, input_domain = space)
main_effects_gp, total_effects_gp, _ = sens_gpbased.compute_effects(num_monte_carlo_points = num_monte_carlo)
main_effects_gp = {ivar: main_effects_gp[ivar][0] for ivar in main_effects_gp}
total_effects_gp = {ivar: total_effects_gp[ivar][0] for ivar in total_effects_gp}

# First order sobol indices:
fig, ax = plt.subplots(figsize=(10, 5))
d = {'GP Monte Carlo':main_effects_gp}
pd.DataFrame(d).plot(kind='bar', ax=ax)
plt.title('First-order Sobol indices - TTI')
plt.ylabel('% of explained output variance')

# Total effects:
fig, ax = plt.subplots(figsize=(10, 5))
d = {'GP Monte Carlo':total_effects_gp}
pd.DataFrame(d).plot(kind='bar', ax=ax)
ax.set_title('Total effects - TTI')
ax.set_ylabel('% of explained output variance')
