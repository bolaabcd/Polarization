from enum import Enum
from functools import partial
import numpy as np

class Update(Enum):
    CLASSIC = "Classic"
    CONFBIAS = "ConfBias"
#####################################
## Update Functions Implementation
#####################################
class Update_Functions():
    def __init__(self,hasDefaults=True,precision: int=4):
        if precision<=0:
            raise ValueError('Precision need to be a positive integer')
        if hasDefaults:
            self.dictionary={
                Update.CLASSIC:self.neighbours_update,
                Update.CONFBIAS:self.neighbours_cb_update,
            }

    def neighbours_update(self,beliefs, inf_graph,**kwargs):
        """Applies the classic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias and
        the beliefs of all the agents' neighbors.

        Equivalent to:

        [blf_ai + np.mean([inf_graph[other, agent] * (blf_aj - blf_ai) for other, blf_aj in enumerate(beliefs) if inf_graph[other, agent] > 0]) for agent, blf_ai in enumerate(beliefs)]

        """
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        return (beliefs @ inf_graph - np.add.reduce(inf_graph) * beliefs) / neighbours + beliefs

    def neighbours_cb_update(self,beliefs, inf_graph,**kwargs):
        """Applies the confirmation-bias update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        confirmation-bias and the beliefs of all the agents' neighbors.

        Equivalent to:
        [blf_ai + np.mean([(1 - np.abs(blf_aj - blf_ai)) * inf_graph[other, agent] * (blf_aj - blf_ai) for other, blf_aj in enumerate(beliefs) if inf_graph[other, agent] > 0]) for agent, blf_ai in enumerate(beliefs)]
        """
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs = inf_graph * (1 - np.abs(diff)) * diff
        return np.add.reduce(infs) / neighbours + beliefs

    def get_function(self,update_type):
        if update_type in self.dictionary:
            return self.dictionary[update_type]
        else:
            raise NotImplementedError(f"Not implemented for {update_type}.")
    def add_function(self,update_type,update_function):
        if update_type in self.dictionary:
            raise ValueError(f"{update_type} already implemented.")
        else:
            self.dictionary[update_type]=update_function

