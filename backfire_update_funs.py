from update_functions import Update_Functions, Enum
from enum import auto
from functools import partial
import numpy as np

class NewUpdate(Enum):
    LINE = "Line"
    MODULUS="Modulus"
    QUADRATIC="Quadratic"
    CUBIC="Cubic"
    MULTIROOT="MultiRoot"#Not used anymore.
    SUPERCUB="SuperCubic"
    CUBICNC="CubicNoConstant"#Just for tests.

class BF_Update_Functions(Update_Functions):
    def __init__(self,precision: int=4):
        if precision<=0:
            raise ValueError('Precision need to be a positive integer')
        super().__init__(precision=precision)
        self.add_function(NewUpdate.LINE,self.neighbours_line_update)
        self.add_function(NewUpdate.MODULUS,self.neighbours_modulus_update)
        self.add_function(NewUpdate.QUADRATIC,self.neighbours_quadratic_update)
        self.add_function(NewUpdate.CUBIC,self.neighbours_cubic_update)
        self.add_function(NewUpdate.CUBICNC,self.neighbours_cubic_nc_update)
        self.add_function(NewUpdate.MULTIROOT,self.neighbours_multiroot_update)
        for i in range(-precision,precision+1):
            self.add_function((NewUpdate.LINE,i/precision),partial(self.neighbours_line_update,rotation_alpha=i/precision))
            self.add_function((NewUpdate.MODULUS,i/precision),partial(self.neighbours_modulus_update,modulus_k=i/precision))
            self.add_function((NewUpdate.QUADRATIC,i/precision),partial(self.neighbours_quadratic_update,quadratic_k=i/precision))
            self.add_function((NewUpdate.CUBIC,i/precision),partial(self.neighbours_cubic_update,cubic_k=i/precision))
            self.add_function((NewUpdate.MULTIROOT,i/precision),partial(self.neighbours_multiroot_update,multi_root_k=i/precision))
            self.add_function((NewUpdate.SUPERCUB,i/precision),partial(self.neighbours_super_update,super_k=i/precision))
            self.add_function((NewUpdate.CUBICNC,i/precision),partial(self.neighbours_cubic_nc_update,cubic_nc_k=i/precision))

    #modulus_k:float=1
    def neighbours_modulus_update(self,beliefs,inf_graph,**kwargs):
        """Applies the inverse-modulus update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        modulus_k=1
        if "modulus_k" in kwargs:
            modulus_k=kwargs["modulus_k"]
        modulus_k=0.5*modulus_k+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        fullMod=np.full(diff.shape,modulus_k)
        #infs = inf_graph * (-np.abs(diff-fullMod)+fullMod)
        sigs=np.where(diff>=0, 1, -1)
        infs=sigs*inf_graph*(-np.abs(np.abs(diff)-fullMod)+fullMod)
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
        #rotation_alpha: kwarg
    def neighbours_line_update(self,beliefs,inf_graph,**kwargs):
        """Applies the rotated-line update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        rotation_alpha=-1
        if "rotation_alpha" in kwargs:
            rotation_alpha=kwargs["rotation_alpha"]
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs = inf_graph * rotation_alpha * diff
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
        
    #quadratic_k:float=2
    def neighbours_quadratic_update(self,beliefs,inf_graph,**kwargs):
        """Applies the inverse-quadratic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        quadratic_k=1
        if "quadratic_k" in kwargs:
            quadratic_k=kwargs["quadratic_k"]
        quadratic_k+=1
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        fullQuad=np.full(diff.shape,quadratic_k)
        #infs = inf_graph * (-diff)*(diff-fullQuad)#-x*(x-k)
        infs=inf_graph * (-diff)*(np.abs(diff)-fullQuad)
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
    #cubic_k:float=0.5
    def neighbours_cubic_update(self,beliefs,inf_graph,**kwargs):
        """Applies the basic cubic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        cubic_k=0
        if "cubic_k" in kwargs:
            cubic_k=kwargs["cubic_k"]
        cubic_k=0.5*cubic_k+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        diff=diff/1.74#Constant
        infs = -inf_graph * diff*(diff-cubic_k)*(diff+cubic_k)*2.6
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
    #multi_root_k:float=0.5)
    def neighbours_multiroot_update(self,beliefs,inf_graph,**kwargs):
        """Applies the basic quadratic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        multi_root_k=0
        if "multi_root_k" in kwargs:
            multi_root_k=kwargs["multi_root_k"]
        multi_root_k=0.5*multi_root_k+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs = inf_graph * diff*(multi_root_k-np.abs(diff))*(1-np.abs(diff))#*27/16
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
    #super_k:float=0
    def neighbours_super_update(self,beliefs,inf_graph,**kwargs):
        """Applies the plus-ultra cubic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        super_k=0
        if "super_k" in kwargs:
            super_k=kwargs["super_k"]
        
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        
        infs = inf_graph * ((super_k+1)*(super_k-1)*diff**3+(-super_k**2+super_k+1)*diff)
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
    #cubic_nc_k:float=0.5
    def neighbours_cubic_nc_update(self,beliefs,inf_graph,**kwargs):
        """Applies the basic quadratic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        cubic_nc_k=0
        if "cubic_nc_k" in kwargs:
            cubic_nc_k=kwargs["cubic_nc_k"]
        cubic_nc_k=0.5*cubic_nc_k+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs = -inf_graph * diff*(diff-cubic_nc_k)*(diff+cubic_nc_k)
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)