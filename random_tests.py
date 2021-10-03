from math import ceil
import matplotlib.pyplot as plt
from Simulation import Simulation
import cli_utils as cli
from belief_states import build_belief, Belief
from influence_graphs import Influence,build_influence
import numpy as np
import os

GLOBAL_SEED = 123456

def get_influence(seed, num_agents, minimum_influence = 0, diagonal_value = None):
    np.random.seed(seed)
    return build_influence(
        Influence.RANDOM, 
        num_agents = num_agents,
        minimum_influence = minimum_influence,
        diagonal_value = diagonal_value
        )

def get_initial_belief(seed, num_agents):
    np.random.seed(seed)
    return build_belief(Belief.RANDOM, num_agents = num_agents)

np.random.seed(GLOBAL_SEED)
def test_final_results_equal(
    container, 
    keys, 
    test_name, 
    number_of_sims = 100000, 
    nagents = 100,
    minimum_influence = 0,
    diagonal_value = None
    ):
    if not os.path.isdir(f"./generated/{GLOBAL_SEED}_equals"):
        os.mkdir(f"./generated/{GLOBAL_SEED}_equals")
    seedsUsed = [(np.random.randint(0, 2**32-1),np.random.randint(0, 2**32-1)) for i in range(number_of_sims)]
    ksUsed = [np.random.rand() for i in range(number_of_sims)]

    initPos = 0
    if os.path.isfile(f"./generated/{GLOBAL_SEED}_equals/{test_name}"):
        arq = open(f"./generated/{GLOBAL_SEED}_equals/{test_name}", "r")
        initPos = int(arq.readline())
        arq.close()
    else:
        arq = open(f"./generated/{GLOBAL_SEED}_equals/{test_name}", "w")
        arq.write("000000\n")
        arq.close()
    
    for i in cli.ProgressRange(number_of_sims, "running"):
        
        if i < initPos:
            continue
        initPos += 1
        rand_bel = get_initial_belief(seedsUsed[i][0],nagents)
        rand_inf = get_influence(
            seedsUsed[i][1],
            nagents,
            minimum_influence,
            diagonal_value
            )
        k = ksUsed[i]
        funs = [container.get_function((key,k)) for key in keys]
        
        final_states = []
        for fun in funs:
            raw = Simulation(rand_bel,rand_inf,fun)
            final_states.append(raw.get_final_state(tolerance = 0))
        
        arq = open(f"./generated/{GLOBAL_SEED}_equals/{test_name}","r+")
        for j in range(len(final_states)):
            out = False
            for l in range(j+1,len(final_states)):
                if np.linalg.norm(final_states[j]-final_states[l]) >  1e-2:
                    # print(final_states[j])
                    # print(final_states[l])
                    # print(rand_inf)
                    # print(rand_bel)
                    print("Difference found!")
                    arq.seek(0,os.SEEK_END)
                    arq.write(f"({k},{seedsUsed[i][0]},{seedsUsed[i][1]})\n")
                    out = True
                    break
            if out:
                break
        arq.seek(0)
        arq.write(str(initPos).zfill(6))
        arq.close()

def plot_differents(
    global_seed, 
    test_name, 
    container, 
    keys, 
    nagents = 100,
    max_time = 1000,
    saveTo = None,
    type = "equals"
    ):
    if not os.path.isfile(f"./generated/{global_seed}_{type}/{test_name}"):
        raise FileNotFoundError("File not found")
    arq = open(f"./generated/{global_seed}_{type}/{test_name}","r")
    lines = arq.readlines()[1:]
    arq.close()

    nlines=ceil(len(lines)*len(keys)/2.0)
    ncols=2
    plt.rcParams.update({'font.size': 16})
    fig=plt.figure(num=1, figsize=(18*ncols,10*nlines))
    plt.subplots_adjust(hspace=0.3)

    for x in cli.ProgressRange(len(lines), "running"):
        case = lines[x]
        tup = eval(case)
        k = tup[0]
        blf_seed = tup[1]
        inf_seed = tup[2]
        blf = get_initial_belief(blf_seed,nagents)
        inf = get_influence(inf_seed,nagents)
        funs = [container.get_function((key,k)) for key in keys]
        completed_sims = []
        for fun in funs:
            raw = Simulation(blf,inf,fun)
            completed_sims.append(raw.run(max_time = max_time, smart_stop=True))

        for i, fin in enumerate(completed_sims):
            plt.subplot(nlines,ncols,len(keys)*x+i+1)                
            plt.plot(fin[1])
            plt.title(f"name={test_name}_{type} fun={keys[i]}")
    if saveTo !=None:
        fig.savefig(saveTo,bbox_inches='tight')
    plt.close()

def test_final_results_value(
    container, 
    keys, 
    test_name, 
    number_of_sims = 100000, 
    nagents = 100,
    minimum_influence = 0,
    diagonal_value = None
):
    if not os.path.isdir(f"./generated/{GLOBAL_SEED}_value"):
        os.mkdir(f"./generated/{GLOBAL_SEED}_value")
    seedsUsed = [(np.random.randint(0, 2**32-1),np.random.randint(0, 2**32-1)) for i in range(number_of_sims)]
    ksUsed = [np.random.rand() for i in range(number_of_sims)]

    initPos = 0
    if os.path.isfile(f"./generated/{GLOBAL_SEED}_value/{test_name}"):
        arq = open(f"./generated/{GLOBAL_SEED}_value/{test_name}", "r")
        initPos = int(arq.readline())
        arq.close()
    else:
        arq = open(f"./generated/{GLOBAL_SEED}_value/{test_name}", "w")
        arq.write("000000\n")
        arq.close()
    
    for i in cli.ProgressRange(number_of_sims, "running"):
        if i < initPos:
            continue
        initPos += 1
        rand_bel = get_initial_belief(seedsUsed[i][0],nagents)
        rand_inf = get_influence(
            seedsUsed[i][1],
            nagents,
            minimum_influence,
            diagonal_value
            )
        k = ksUsed[i]
        funs = [container.get_function((key,k)) for key in keys]
        
        final_states = []
        for fun in funs:
            raw = Simulation(rand_bel,rand_inf,fun)
            final_states.append(raw.get_final_state(tolerance = 0))
        
        predicted = []
        for fun in funs:
            indegrees = np.array([np.count_nonzero(rand_inf[:, i]) for i in range(len(rand_bel))])
            infs = rand_inf.copy()
            np.fill_diagonal(infs,1)
            influences = np.array([np.prod(infs[i,:]) for i in range(len(rand_bel))])
            
            if np.all(influences != 0):
                infs[infs == 0] = 1
                influences = np.array([np.prod(infs[i,:]) for i in range(len(rand_bel))])
                predicted.append(np.sum(indegrees*influences*rand_bel)/np.sum(indegrees*influences))
            elif np.all(influences == 0):
                predicted.append(rand_bel)
            else:
                rand_bel = np.array(rand_bel)
                predicted.append(rand_bel[influences!=0][0])

        arq = open(f"./generated/{GLOBAL_SEED}_value/{test_name}","r+")
        for j in range(len(final_states)):
            if np.linalg.norm(final_states[j] - predicted[j])>1e-3:
                print("Difference found!")
                arq.seek(0,os.SEEK_END)
                arq.write(f"({k},{seedsUsed[i][0]},{seedsUsed[i][1]})\n")
                break
        arq.seek(0)
        arq.write(str(initPos).zfill(6))
        arq.close()

