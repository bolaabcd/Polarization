import math
from enum import Enum
import numpy as np

######################################################
## Parameters for the Belief states
######################################################

## default number of agents
NUM_AGENTS = 100


############################################
## Representing belief states implementation
############################################

class Belief(Enum):
    UNIFORM = 0
    MILD = 1
    EXTREME = 2
    TRIPLE = 3
    RANDOM = 4

def build_belief(belief_type: Belief, num_agents=NUM_AGENTS, **kwargs):
    """Evenly distributes the agents beliefs into subgroups.
    
    """
    if belief_type is Belief.MILD:
        middle = math.ceil(num_agents / 2)
        return [0.2 + 0.2 * i / middle if i < middle else 0.6 + 0.2 * (i - middle) / (num_agents - middle) for i in range(num_agents)]
    if belief_type is Belief.EXTREME:
        middle = math.ceil(num_agents / 2)
        return [0.2 * i / middle if i < middle else 0.8 + 0.2 * (i - middle) / (num_agents - middle) for i in range(num_agents)]
    if belief_type is Belief.TRIPLE:
        beliefs = [0.0] * num_agents
        first_third = num_agents // 3
        middle_third = math.ceil(num_agents * 2 / 3) - first_third
        last_third = num_agents - middle_third - first_third
        offset = 0
        for i, segment in enumerate((first_third, middle_third, last_third)):
            for j in range(segment):
                beliefs[j+offset] = 0.2 * j / segment + (0.4 * i)
            offset += segment
        return beliefs
    if belief_type is Belief.UNIFORM:
        return [i/(num_agents - 1) for i in range(num_agents)]
    if belief_type is Belief.RANDOM:
        return np.ndarray.tolist(np.random.uniform(0,1,num_agents))
