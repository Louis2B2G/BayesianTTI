import os

import numpy as np
import pandas as pd
from tqdm.notebook import trange

from tti_explorer import config, utils
from tti_explorer.case import simulate_case, CaseFactors
from tti_explorer.contacts import EmpiricalContactsSimulator
from tti_explorer.strategies import TTIFlowModel, RETURN_KEYS

rng = np.random.RandomState(0)

from tti_explorer.strategies import TTIFlowModel


# parameters
n_cases = 10000
name = 'S4_test_based_TTI'

case_config = config.get_case_config("delve")
contacts_config = config.get_contacts_config("delve")
policy_config = config.get_strategy_configs("delve", name)[name]


configs = [case_config, contacts_config, policy_config]


def simulation(theta, config_details):
    """
        theta: list in R^d where d is the number of parameters
        config_details: dictionary where config_details[i] contains the index corresponding
                        to the config file in configs and the name associated to theta (for changing the config)
    """
    
    for i, val in enumerate(theta):
        index = config_details[i]["index"]
        config_name = config_details[i]["name"]
        configs[index][config_name] = val
    
    factor_config = utils.get_sub_dictionary(configs[2], config.DELVE_CASE_FACTOR_KEYS)
    strategy_config = utils.get_sub_dictionary(configs[2], config.DELVE_STRATEGY_FACTOR_KEYS)
    
    rng = np.random.RandomState(42)
    simulate_contacts = EmpiricalContactsSimulator(over18, under18, rng)
    tti_model = TTIFlowModel(rng, **strategy_config)
    
    outputs = list()

    for _ in trange(n_cases):
        case = simulate_case(rng, **configs[0])
        case_factors = CaseFactors.simulate_from(rng, case, **factor_config)
        contacts = simulate_contacts(case, **configs[1])
        res = tti_model(case, contacts, case_factors)
        outputs.append(res)

    effective_R = pd.DataFrame(outputs).mean(0).loc[RETURN_KEYS.reduced_r]
    return effective_R
   
   
if __name__ == "main":
	# example with theta only having one value
	theta = [0]
	config_details = {0 : {"name": "p_under18", "index": 0}}
	simulation(theta, config_details)