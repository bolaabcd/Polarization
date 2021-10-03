from polarization_measure import Polarization_Measure
import numpy as np
#from update_functions import make_update_fn, Update
from update_functions import Update, Update_Functions
from Esteban_Ray_polarization import Esteban_Ray_polarization
import warnings

#######################################
## Main Simulation Class Implementation
#######################################
MAX_TIME=200

class Simulation:
    def __init__(self, belief_vec, inf_graph, update_fun, pol_instance : Polarization_Measure=Esteban_Ray_polarization(), **kwargs):
        self.belief_vec = np.array(belief_vec)
        self.inf_graph = inf_graph
        self.update_fun = update_fun
        self.pol_instance = pol_instance
        self.kwargs=kwargs
        
    def __iter__(self):
        return self

    def __next__(self):
        result = (self.belief_vec, self.pol_instance.pol_measure(self.belief_vec))
        self.belief_vec = self.update_fun(self.belief_vec, self.inf_graph,**self.kwargs)
        return result

    def run(self, max_time=100, smart_stop=True):
        """Runs the current Simulation setup.

        For each time step, calculate the polarization value and the belief
        state. If `smart_stop` is `True`, the simulation stops when the
        belief state does not change from one time step to the next or if
        `max_time` is reached.

        Args:
          - max_time (int, default 100): The time step to stop the simulation.
          - smart_stop (Boolean, default True): Stops the simulation if the
            belief state stabilizes or `max_time` is reached.

        Returns:
          A tuple of NumPy Arrays with each polarization value, its
          corresponding  belief state, and the last polarization value.
          (pol_history, blf_history, pol_history[-1]).
        """
        belief_history = []
        pol_history = []
        for _, (belief_vec_state, pol_state) in zip(range(max_time), self):
            # Stop if a stable state is reached
            if smart_stop and belief_history and np.array_equal(belief_history[-1], belief_vec_state):
                break
            belief_history.append(belief_vec_state)
            pol_history.append(pol_state)
        return (np.array(pol_history), np.array(belief_history), pol_history[-1])
    def get_final_state(self,max_time=50000 ,tolerance=1e-6):
        previous_belief=np.array([])
        for _, (belief_vec_state, pol_state) in zip(range(max_time), self):
            if (previous_belief.size> 0) and np.linalg.norm(previous_belief-belief_vec_state)<=tolerance:
                return belief_vec_state
            previous_belief=belief_vec_state
        return belief_vec_state
        