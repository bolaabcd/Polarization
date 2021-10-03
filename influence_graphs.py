import math
from enum import Enum
import numpy as np
from belief_states import NUM_AGENTS

#######################################
## Parameters for influence graphs
#######################################

## for clique influence-graph: belief value of all agents on a clique influence graph
CLIQUE_INF_VALUE = 0.5

## for 2_groups_disconnected influence-graph: belief value of all agents that can communicate in a 2 groups_disconnected influence graph
GROUPS_DISCONNECTED_INF_VALUE = 0.5

## for 2_groups_faint ineraction-function: belief value of all agents that can strongly communicate in a 2 groups faintly connected influence graph
GROUPS_FAINTLY_INF_VALUE_STRONG = 0.5
## for 2_groups_faint ineraction-function: belief value of all agents that can weakly communicate in a 2 groups faintly connected influence graph
GROUPS_FAINTLY_INF_VALUE_WEAK = 0.1

## for 2_influencers_balanced influence-graph: level of influence both influencers exert on all others
INFLUENCERS_BALANCED_OUTGOING_BOTH = 0.6
## for 2_influencers_balanced influence-graph: level of influence both influencers receive from all others
INFLUENCERS_BALANCED_INCOMING_BOTH = 0.0
## for 2_influencers_balanced influence-graph: level of influence all other agents exert on all others
INFLUENCERS_BALANCED_OTHERS = 0.1

## for 2_influencers_unbalanced influence-graph: level of influence agent 0 exerts on all others
INFLUENCERS_UNBALANCED_OUTGOING_FIRST = 0.8
## for 2_influencers_unbalanced influence-graph: level of influence agent n-1 exerts on all others
INFLUENCERS_UNBALANCED_OUTGOING_SECOND = 0.5
## for 2_influencers_unbalanced influence-graph: level of influence agent 0 receives from all others
INFLUENCERS_UNBALANCED_INCOMING_FIRST = 0.1
## for 2_influencers_unbalanced influence-graph: level of influence agent n-1 receives from all others
INFLUENCERS_UNBALANCED_INCOMING_SECOND = 0.1
## for 2_influencers_unbalanced influence-graph: level of influence all other agents exert on all others
INFLUENCERS_UNBALANCED_OTHERS = 0.2

## for circular influence-graph: belief value of all agents on a circular influence graph
CIRCULAR_INF_VALUE = 0.5

## for random influence graph: minimum influence value
MIN_INF = 0


#####################################
## Influence graphs implementation
#####################################

def build_inf_graph_clique(num_agents, belief_value):
    """Returns the influence graph for "clique" scenario."""
    return np.full((num_agents, num_agents), belief_value)

def build_inf_graph_2_groups_disconnected(num_agents, belief_value):
    """Returns the influence graph for for "disconnected" scenario."""
    inf_graph = np.zeros((num_agents, num_agents))
    middle = math.ceil(num_agents / 2)
    inf_graph[:middle, :middle] = belief_value
    inf_graph[middle:, middle:] = belief_value
    return inf_graph

def build_inf_graph_2_groups_faint(num_agents, weak_belief_value, strong_belief_value):
    """Returns the influence graph for for "faintly-connected" scenario."""
    inf_graph = np.full((num_agents, num_agents), weak_belief_value)
    middle = math.ceil(num_agents / 2)
    inf_graph[:middle, :middle] = strong_belief_value
    inf_graph[middle:, middle:] = strong_belief_value
    return inf_graph

def build_inf_graph_2_influencers_balanced(num_agents, influencers_incoming_value, influencers_outgoing_value, others_belief_value):
    """Returns the influence graph for for "balanced 2-influencers" scenario."""
    inf_graph = np.full((num_agents, num_agents), others_belief_value)
    ## Sets the influence of agent 0 on all others
    inf_graph[0, :-1] = influencers_outgoing_value
    ## Sets the influence of agent n-1 on all others
    inf_graph[-1, 1:] = influencers_outgoing_value
    ## Sets the influence of all other agents on agent 0.
    inf_graph[1:,0] = influencers_incoming_value
    ## Sets the influence of all other agents on agent n-1.
    inf_graph[:-1, -1] = influencers_incoming_value
    return inf_graph

def build_inf_graph_2_influencers_unbalanced(num_agents, influencers_outgoing_value_first, influencers_outgoing_value_second, influencers_incoming_value_first, influencers_incoming_value_second, others_belief_value):
    """Returns the influence graph for for "unbalanced 2-influencers" scenario."""
    inf_graph = np.full((num_agents,num_agents), others_belief_value)
    ## Sets the influence of agent 0 on all others
    inf_graph[0, :-1] = influencers_outgoing_value_first
    ## Sets the influence of agent n-1 on all others
    inf_graph[-1, 1:] = influencers_outgoing_value_second
    ## Sets the influence of all other agents on agent 0.
    inf_graph[1:, 0] = influencers_incoming_value_first
    ## Sets the influence of all other agents on agent n-1.
    inf_graph[:-1, -1] = influencers_incoming_value_second
    return inf_graph

def build_inf_graph_circular(num_agents, value):
    """Returns the imfluemce graph for "circular influence" scenario."""
    inf_graph = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        inf_graph[i, i] = 1.0
        inf_graph[i, (i+1) % num_agents] = value
    return inf_graph

def build_inf_graph_random(num_agents,diagonal_value=None,minimum_influence=MIN_INF):
    conects=np.random.random_integers(0,1,(num_agents,num_agents))
    rands=np.random.uniform(minimum_influence,1,(num_agents,num_agents))
    ans=conects*rands
    if diagonal_value!=None:
        np.fill_diagonal(ans,diagonal_value)
    return ans

class Influence(Enum):
    CLIQUE = 0
    GROUP_2_DISCONECTED = 1
    GROUP_2_FAINT = 2
    INFLUENCERS_2_BALANCED = 3
    INFLUENCERS_2_UNBALANCED = 4
    CIRCULAR = 5
    RANDOM = 6

def build_influence(
        inf_type: Influence,
        num_agents=NUM_AGENTS,
        *,
        weak_belief=GROUPS_FAINTLY_INF_VALUE_WEAK,
        strong_belief=GROUPS_FAINTLY_INF_VALUE_STRONG,
        general_belief=None,
        influencer_incoming_belief=None,
        influencer_outgoing_belief=None,
        influencer2_incoming_belief=INFLUENCERS_UNBALANCED_INCOMING_SECOND,
        influencer2_outgoing_belief=INFLUENCERS_UNBALANCED_OUTGOING_SECOND,
        minimum_influence = MIN_INF,
        diagonal_value = None):
    """Builds the initial influence graph according to the `inf_type`.

    Helper function when iterating over the `Influence` enum. The default values
    are the constants defined at the beginning of the polarization module.
    """
    if inf_type is Influence.CLIQUE:
        if general_belief is None:
            general_belief = CLIQUE_INF_VALUE
        return build_inf_graph_clique(num_agents, general_belief)
    if inf_type is Influence.GROUP_2_DISCONECTED:
        if general_belief is None:
            general_belief = GROUPS_DISCONNECTED_INF_VALUE
        return build_inf_graph_2_groups_disconnected(num_agents, general_belief)
    if inf_type is Influence.GROUP_2_FAINT:
        return build_inf_graph_2_groups_faint(num_agents, weak_belief, strong_belief)
    if inf_type is Influence.INFLUENCERS_2_BALANCED:
        if general_belief is None:
            general_belief = INFLUENCERS_BALANCED_OTHERS
        if influencer_incoming_belief is None:
            influencer_incoming_belief = INFLUENCERS_BALANCED_INCOMING_BOTH
        if influencer_outgoing_belief is None:
            influencer_outgoing_belief = INFLUENCERS_BALANCED_OUTGOING_BOTH
        return build_inf_graph_2_influencers_balanced(num_agents, influencer_incoming_belief, influencer_outgoing_belief, general_belief)
    if inf_type is Influence.INFLUENCERS_2_UNBALANCED:
        if general_belief is None:
            general_belief = INFLUENCERS_UNBALANCED_OTHERS
        if influencer_incoming_belief is None:
            influencer_incoming_belief = INFLUENCERS_UNBALANCED_INCOMING_FIRST
        if influencer2_incoming_belief is None:
            influencer2_incoming_belief = INFLUENCERS_UNBALANCED_INCOMING_SECOND
        if influencer_outgoing_belief is None:
            influencer_outgoing_belief = INFLUENCERS_UNBALANCED_OUTGOING_FIRST
        if influencer2_outgoing_belief is None:
            influencer2_outgoing_belief = INFLUENCERS_UNBALANCED_OUTGOING_SECOND
        return build_inf_graph_2_influencers_unbalanced(num_agents, influencer_outgoing_belief, influencer2_outgoing_belief, influencer_incoming_belief, influencer2_incoming_belief, general_belief)
    if inf_type is Influence.CIRCULAR:
        if general_belief is None:
            general_belief = CIRCULAR_INF_VALUE
        return build_inf_graph_circular(num_agents, general_belief) 
    if inf_type is Influence.RANDOM:
        return build_inf_graph_random(
            num_agents,
            diagonal_value = diagonal_value,
            minimum_influence = minimum_influence
            )
    raise Exception('inf_type not recognized. Expected an `Influence`')
