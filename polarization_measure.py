#Basic interface for polarization measures
class Polarization_Measure:
    def __init__(self):
        pass
    #All polarization measure classes should have this method
    def pol_measure(self,belief_state):
        pass