from update_functions import Update_Functions, Enum
import numpy as np

class NewUpdate(Enum):
    LINE = "Line"
    MODULUS="Modulus"
    QUADRATIC="Quadratic"
    CUBIC="Cubic"
    MULTIROOT="MultiRoot"#Not used anymore.
    INTERPOLATED="InterpolatedCubic"

class BF_Update_Functions(Update_Functions):
    def __init__(self):
        super().__init__()
        self.add_function(NewUpdate.LINE,self.neighbours_line_update)
        self.add_function(NewUpdate.MODULUS,self.neighbours_modulus_update)
        self.add_function(NewUpdate.QUADRATIC,self.neighbours_quadratic_update)
        self.add_function(NewUpdate.CUBIC,self.neighbours_cubic_update)
        self.add_function(NewUpdate.MULTIROOT,self.neighbours_multiroot_update)
        self.add_function(NewUpdate.INTERPOLATED,self.neighbours_super_update)

    def neighbours_modulus_update(self,beliefs,inf_graph, rat_graph,**kwargs):
        """Applies the inverse-modulus update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        
        rat_graph=rat_graph.copy()
        rat_graph=0.5*rat_graph+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        sigs=np.where(diff>=0, 1, -1)
        infs=sigs*inf_graph*(-np.abs(np.abs(diff)-rat_graph)+rat_graph)
        preAns=np.add.reduce(infs) / neighbours
        preAns+=beliefs
        return np.clip(preAns,0,1)

    def neighbours_line_update(self,beliefs,inf_graph, rat_graph,**kwargs):
        """Applies the rotated-line update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        rat_graph=rat_graph.copy()
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs = inf_graph * rat_graph * diff
        preAns=np.add.reduce(infs) / neighbours
        preAns+=beliefs
        return np.clip(preAns,0,1)
        
    def neighbours_quadratic_update(self,beliefs,inf_graph, rat_graph,**kwargs):
        """Applies the inverse-quadratic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        rat_graph=rat_graph.copy().T
        rat_graph+=1
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs=inf_graph * (-diff)*(np.abs(diff)-rat_graph)
        preAns=np.add.reduce(infs) / neighbours
        preAns/=2
        preAns+=beliefs
        return np.clip(preAns,0,1)

    def neighbours_cubic_update(self,beliefs,inf_graph, rat_graph,**kwargs):
        """Applies the basic cubic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        rat_graph=rat_graph.copy()
        rat_graph=-rat_graph-1
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs = -inf_graph * diff*(diff-rat_graph)*(diff+rat_graph)
        preAns=np.add.reduce(infs) / neighbours
        preAns/=4
        preAns+=beliefs
        return np.clip(preAns,0,1)
    
    def neighbours_multiroot_update(self,beliefs,inf_graph, rat_graph,**kwargs):
        """Applies the basic quadratic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        rat_graph=rat_graph.copy()
        rat_graph=0.5*rat_graph+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs = inf_graph * diff*(rat_graph-np.abs(diff))*(1-np.abs(diff))#*27/16
        preAns=np.add.reduce(infs) / neighbours
        preAns+=beliefs
        return np.clip(preAns,0,1)

    def neighbours_super_update(self,beliefs,inf_graph, rat_graph,**kwargs):
        """Applies the plus-ultra cubic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        rat_graph=rat_graph.copy()
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        
        infs = inf_graph * ((rat_graph+1)*(rat_graph-1)*diff**3+(-rat_graph**2+rat_graph+1)*diff)
        preAns=np.add.reduce(infs) / neighbours
        # np.nan_to_num(preAns,copy=False)
        preAns/=1.5
        preAns+=beliefs
        return np.clip(preAns,0,1)