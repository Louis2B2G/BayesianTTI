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


configs = {"case_config": case_config, "contacts_config": contacts_config, "policy_config": policy_config}


def simulation(theta, config_details):
    """
        theta: list in R^d where d is the number of parameters
        config_details: dictionary where config_details[i] contains the index corresponding
                        to the config file in configs and the name associated to theta (for changing the config)
    """
    
    for i, val in enumerate(theta):
        config_type = config_details[i]["config"]
        config_name = config_details[i]["name"]
        configs[config_type][config_name] = val
    
    factor_config = utils.get_sub_dictionary(configs["policy_config"], config.DELVE_CASE_FACTOR_KEYS)
    strategy_config = utils.get_sub_dictionary(configs["policy_config"], config.DELVE_STRATEGY_FACTOR_KEYS)
    
    rng = np.random.RandomState(42)
    simulate_contacts = EmpiricalContactsSimulator(over18, under18, rng)
    tti_model = TTIFlowModel(rng, **strategy_config)
    
    outputs = list()

    for _ in trange(n_cases):
        case = simulate_case(rng, **configs["case_config"])
        case_factors = CaseFactors.simulate_from(rng, case, **factor_config)
        contacts = simulate_contacts(case, **configs["contacts_config"])
        res = tti_model(case, contacts, case_factors)
        outputs.append(res)

    effective_R = pd.DataFrame(outputs).mean(0).loc[RETURN_KEYS.reduced_r]
    return effective_R   
   
if __name__ == "main":
	# example with theta having two values to optimize for
	# example with theta having two values to optimize for
	theta = [0, 1]

	config_details = {0 : {"name": "p_under18", "config": "case_config"}, 
                 1 : {"name": "compliance", "config": "policy_config"}}

	simulation(theta, config_details)