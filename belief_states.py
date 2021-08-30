import math
from enum import Enum


######################################################
## Parameters for the Belief states
######################################################

## number of agents
NUM_AGENTS = 100

## for consensus belief-function: belief value for the consensus belief state
CONSENSUS_VALUE = 0.5

## Values for the old belief configurations:
## -----------------------------------------
## for mildly-polarized belief-function: belief value for the upper end of the low pole of a mildly polarized belief state
LOW_POLE = 0.25
## for mildly-polarized belief-function: belief value for the lower end of the high pole of a mildly polarized belief state
HIGH_POLE = 0.75
## for mildly-polarized belief-function: step of belief change from agent to agent in mildly polarized belief state
BELIEF_STEP = 0.01

############################################
## Representing belief states implementation
############################################

class Belief(Enum):
    UNIFORM = 0
    MILD = 1
    EXTREME = 2
    TRIPLE = 3
#    CONSENSUS = 4

## Current representation

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
