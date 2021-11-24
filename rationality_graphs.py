from enum import Enum
import numpy as np
from belief_states import NUM_AGENTS

#######################################
## Parameters for rationality graphs
#######################################

## Default rationality value of agents
DEFAULT_RAT_VALUE = 0


#####################################
## Influence graphs implementation
#####################################

def build_rat_graph_constant(num_agents,rationality_value):
    """Returns the rationality graph in which all agents have the same rationality value."""
    return np.full((num_agents,num_agents),rationality_value)

#rationality_values should be a list of rationality values of each agent
def build_rat_graph_per_agent(num_agents, rationality_values: list):
    """Returns the rationality graph for rationality innerent to agents."""
    return np.full((num_agents, num_agents), 0) + np.array(rationality_values)[np.newaxis,:]


def build_inf_graph_random(num_agents):
    return np.random.uniform(-1,1,(num_agents,num_agents))

class Rationality(Enum):
    CONSTANT = 0
    PER_AGENT = 1
    RANDOM = 2

def build_rat_graph(
        rat_type: Rationality,
        num_agents=NUM_AGENTS,
        rationality_value=None,
        rationality_values=None,
        ):
    """Builds the initial rationality graph according to the `rat_type`.

    Helper function when iterating over the `rationality` enum. The default values
    are the constants defined at the beginning of the polarization module.
    """
    if rat_type is Rationality.CONSTANT:
        if rationality_value is None:
            rationality_value = DEFAULT_RAT_VALUE
        return build_rat_graph_constant(num_agents, DEFAULT_RAT_VALUE)
    if rat_type is Rationality.PER_AGENT:
        if rationality_values is None:
            rationality_values = [DEFAULT_RAT_VALUE for i in range(num_agents)]
        return build_rat_graph_per_agent(num_agents, rationality_values)
    if rat_type is Rationality.RANDOM:
        return build_inf_graph_random(num_agents)
    raise Exception('rat_type not recognized. Expected an `rationality`')
