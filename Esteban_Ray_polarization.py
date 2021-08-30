import numpy as np
import math
from functools import partial
from polarization_measure import Polarization_Measure

class Esteban_Ray_polarization(Polarization_Measure):
    def __init__(self, num_bins=5,alpha=1.6,K=1000) -> None:
        self.K=K
        self.num_bins=num_bins
        self.alpha=alpha
    def _belief_2_dist(self,belief_vec):
        """Takes a belief state `belief_vec` and discretizes it into a fixed
        number of bins.
        """
        # stores labels of bins
        # the value of a bin is the medium point of that bin
        bin_labels = [(i + 0.5)/self.num_bins for i in range(self.num_bins)]

        # stores the distribution of labels
        bin_prob = [0] * self.num_bins
        # for all agents...
        for belief in belief_vec:
            # computes the bin into which the agent's belief falls
            bin_ = math.floor(belief * self.num_bins)
            # treats the extreme case in which belief is 1, putting the result in the right bin.
            if bin_ == self.num_bins:
                bin_ = self.num_bins - 1
            # updates the frequency of that particular belief
            bin_prob[bin_] += 1 / len(belief_vec)
        # bundles into a matrix the bin_labels and bin_probabilities.
        dist = np.array([bin_labels,bin_prob])
        # returns the distribution.
        return dist

    def _make_belief_2_dist_func(self):
        """Returns a function that discretizes a belief state into a `num_bins`
        number of bins.
        """
        __belief_2_dist = partial(self.belief_2_dist, num_bins=self.num_bins)
        __belief_2_dist.__name__ = self.belief_2_dist.__name__
        __belief_2_dist.__doc__ = self.belief_2_dist.__doc__
        return __belief_2_dist

    def _pol_ER(self,dist):
        """Computes the Esteban-Ray polarization of a distribution.
        """
        # recover bin labels
        bin_labels = dist[0]
        # recover bin probabilities
        bin_prob = dist[1]

        diff = np.ones((len(bin_labels), 1)) @ bin_labels[np.newaxis]
        diff = np.abs(diff - np.transpose(diff))
        pol = (bin_prob ** (1 + self.alpha)) @ diff @ bin_prob
        # scales the measure by the constant K, and returns it.
        return self.K * pol

    def _make_pol_er_func(self):
        """Returns a function that computes the Esteban-Ray polarization of a
        distribution with set parameters `alpha` and `K`
        """
        __pol_ER = partial(self._pol_ER, alpha=self.alpha, K=self.K)
        __pol_ER.__name__ = self._pol_ER.__name__
        __pol_ER.__doc__ = self._pol_ER.__doc__
        return __pol_ER

    def _pol_ER_discretized(self,belief_state):
        """Discretize belief state as necessary for computing Esteban-Ray
        polarization and computes it.
        """
        return self._pol_ER(self._belief_2_dist(belief_state))

    def make_pol_er_discretized_func(self):
        """Returns a function that computes the Esteban-Ray polarization of a
        belief state, discretized into a `num_bins` number of bins, with set
        parameters `alpha` and `K`.
        """
        __pol_ER_discretized = partial(self._pol_ER_discretized, alpha=self.alpha, K=self.K, num_bins=self.num_bins)
        __pol_ER_discretized.__name__ = self._pol_ER_discretized.__name__
        __pol_ER_discretized.__doc__ = self._pol_ER_discretized.__doc__
        return __pol_ER_discretized

    #All polarization classes should have this method
    def pol_measure(self,belief_state):
        return self._pol_ER_discretized(belief_state)
