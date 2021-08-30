from update_functions import Update_Functions, Enum
from enum import auto
from functools import partial
import numpy as np

class NewUpdate(Enum):
    INVMOD="Modulus"
    INVQUAD="Quadratic"
    CUBIC="Cubic"
    BFISOLATED="IsolatedBF"
    SUPERCUB="SuperCubic"
    CUBICNC="CubicNoConstant"

class BF_Update_Functions(Update_Functions):
    def __init__(self,precision: int=4):
        if precision<=0:
            raise ValueError('Precision need to be a positive integer')
        super().__init__(precision=precision)
        self.add_function(NewUpdate.INVMOD,self.inverse_modulus_update)
        self.add_function(NewUpdate.INVQUAD,self.inverse_quadratic_update)
        self.add_function(NewUpdate.CUBIC,self.cubic_backfire)
        self.add_function(NewUpdate.BFISOLATED,self.backfire_isolated)
        for i in range(-precision,precision+1):
            self.add_function((NewUpdate.INVMOD,i/precision),partial(self.inverse_modulus_update,modulus_factor=i/precision))
            self.add_function((NewUpdate.INVQUAD,i/precision),partial(self.inverse_quadratic_update,quadratic_factor=i/precision))
            self.add_function((NewUpdate.CUBIC,i/precision),partial(self.cubic_backfire,root_backfire_treshold=i/precision))
            self.add_function((NewUpdate.BFISOLATED,i/precision),partial(self.backfire_isolated,isolated_backfire_treshold=i/precision))
            self.add_function((NewUpdate.SUPERCUB,i/precision),partial(self.ultra_cubic_backfire,plus_ultra_k=i/precision))
            self.add_function((NewUpdate.CUBICNC,i/precision),partial(self.cubic_nc_backfire,root_nc_backfire_treshold=i/precision))

    #modulus_factor:float=1
    def inverse_modulus_update(self,beliefs,inf_graph,**kwargs):
        """Applies the inverse-modulus update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        modulus_factor=1
        if "modulus_factor" in kwargs:
            modulus_factor=kwargs["modulus_factor"]
        modulus_factor=0.5*modulus_factor+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        fullMod=np.full(diff.shape,modulus_factor)
        #infs = inf_graph * (-np.abs(diff-fullMod)+fullMod)
        sigs=np.where(diff>=0, 1, -1)
        infs=sigs*inf_graph*(-np.abs(np.abs(diff)-fullMod)+fullMod)
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
    #quadratic_factor:float=2
    def inverse_quadratic_update(self,beliefs,inf_graph,**kwargs):
        """Applies the inverse-quadratic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        quadratic_factor=1
        if "quadratic_factor" in kwargs:
            quadratic_factor=kwargs["quadratic_factor"]
        quadratic_factor+=1
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        fullQuad=np.full(diff.shape,quadratic_factor)
        #infs = inf_graph * (-diff)*(diff-fullQuad)#-x*(x-k)
        infs=inf_graph * (-diff)*(np.abs(diff)-fullQuad)
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
    #root_backfire_treshold:float=0.5
    def cubic_backfire(self,beliefs,inf_graph,**kwargs):
        """Applies the basic cubic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        root_backfire_treshold=0
        if "root_backfire_treshold" in kwargs:
            root_backfire_treshold=kwargs["root_backfire_treshold"]
        root_backfire_treshold=0.5*root_backfire_treshold+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        diff=diff/1.74#Constant
        infs = -inf_graph * diff*(diff-root_backfire_treshold)*(diff+root_backfire_treshold)*2.6
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
    #isolated_backfire_treshold:float=0.5)
    def backfire_isolated(self,beliefs,inf_graph,**kwargs):
        """Applies the basic quadratic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        isolated_backfire_treshold=0
        if "isolated_backfire_treshold" in kwargs:
            isolated_backfire_treshold=kwargs["isolated_backfire_treshold"]
        isolated_backfire_treshold=0.5*isolated_backfire_treshold+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs = inf_graph * diff*(isolated_backfire_treshold-np.abs(diff))*(1-np.abs(diff))#*27/16
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
    #plus_ultra_k:float=0
    #BROKEN
    def ultra_cubic_backfire(self,beliefs,inf_graph,**kwargs):
        """Applies the plus-ultra cubic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        plus_ultra_k=0
        if "plus_ultra_k" in kwargs:
            plus_ultra_k=kwargs["plus_ultra_k"]
        
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        
        infs = inf_graph * ((plus_ultra_k+1)*(plus_ultra_k-1)*diff**3+(-plus_ultra_k**2+plus_ultra_k+1)*diff)
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)
    #root_nc_backfire_treshold:float=0.5
    def cubic_nc_backfire(self,beliefs,inf_graph,**kwargs):
        """Applies the basic quadratic update function as matrix multiplication.
        
        For each agent, update their beliefs factoring the authority bias,
        the confirmation-backfire factor and the beliefs of all the agents' neighbors.
        """
        root_nc_backfire_treshold=0
        if "root_nc_backfire_treshold" in kwargs:
            root_nc_backfire_treshold=kwargs["root_nc_backfire_treshold"]
        root_nc_backfire_treshold=0.5*root_nc_backfire_treshold+0.5
        neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
        diff = np.ones((len(beliefs), 1)) @  np.asarray(beliefs)[np.newaxis]
        diff = np.transpose(diff) - diff
        infs = -inf_graph * diff*(diff-root_nc_backfire_treshold)*(diff+root_nc_backfire_treshold)
        preAns=np.add.reduce(infs) / neighbours + beliefs
        return np.clip(preAns,0,1)