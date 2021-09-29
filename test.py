from math import modf
from Simulation import Simulation
import cli_utils as cli
from backfire_update_funs import BF_Update_Functions, NewUpdate
from belief_states import build_belief, Belief
from influence_graphs import Influence,build_influence
import matplotlib.pyplot as plt
import numpy as np

NAGENTS=3
INITIAL_POSITION=0
SAVE_POSITION=1000000
test_converges=True


#Creating function container:
fun_container=BF_Update_Functions()

#sym_seed=np.random.randint(0,2**32-1)
#np.random.seed(sym_seed)
my_seed=13062002
np.random.seed(my_seed)
for i in cli.ProgressRange(100000, "running"):
    k=np.random.rand()
    #nagents=np.random.randint(1,100)
    nagents=NAGENTS
    randbel=build_belief(Belief.RANDOM,nagents)
    randinf=build_influence(Influence.RANDOM,nagents)

    if i<INITIAL_POSITION:
        continue
    
    if i==SAVE_POSITION:
        bel=open(f"generated/{nagents}_agents_k{k}_{my_seed}_{i}_belief.bin","w")
        np.ndarray.tofile(np.array(randbel),bel)
        bel.close()
        infl=open(f"generated/{nagents}_agents_k{k}_{my_seed}_{i}_influence.bin","w")
        np.ndarray.tofile(randinf,infl)
        infl.close()

    quadRaw=Simulation(
        randbel,
        randinf,
        fun_container.get_function((NewUpdate.QUADRATIC,k))
    )

    modRaw=Simulation(
        randbel,
        randinf,
        fun_container.get_function((NewUpdate.MODULUS,k))
    )

    cubNcRaw=Simulation(
        randbel,
        randinf,
        fun_container.get_function((NewUpdate.CUBICNC,k))
    )

    quadFin=quadRaw.get_final_state(tolerance=0*1e-10)
    modFin=modRaw.get_final_state(tolerance=0*1e-10)
    cubNcFin=cubNcRaw.get_final_state(tolerance=0*1e-10)
    
    #if i==25:
    #    print(quadFin)
    #    print(modFin)
    #    print(cubNcFin)
    
    if test_converges:
        alleq=True
        #print(quadFin)
        #print(modFin)
        #print(cubNcFin)
        #print(quadFin==quadFin[0])
        #if not np.all(quadFin==quadFin[0]):
        if not np.all(np.abs(quadFin-quadFin[0])<1e-5):
            alleq=False
        if not (np.all(np.abs(modFin-modFin[0])<1e-5)==alleq):
            raise RuntimeError("Different final results.")
        if not (np.all(np.abs(cubNcFin-cubNcFin[0])<1e-5)==alleq):
            raise RuntimeError("Different final results.")
        
    elif nagents==2 or nagents==100:
        if np.linalg.norm(quadFin-modFin)>1e-2:
            raise RuntimeError("Error")

        if np.linalg.norm(quadFin-cubNcFin)>1e-2:
            raise RuntimeError("Error")
        
        if np.linalg.norm(modFin-cubNcFin)>1e-2:
            raise RuntimeError("Error")
        
        if i==24 and nagents==3:
            print(quadFin)
            print(modFin)
            print(cubNcFin)
    else:
        print(quadFin[0],end=" ")
        print(modFin[0],end=" ")
        print(cubNcFin[0])
