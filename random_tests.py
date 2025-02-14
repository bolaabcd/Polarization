from math import ceil
from typing import final
import matplotlib.pyplot as plt
from Simulation import Simulation
import cli_utils as cli
from belief_states import build_belief, Belief
from influence_graphs import Influence,build_influence
from rationality_graphs import Rationality,build_rat_graph
import numpy as np
import os

GLOBAL_SEED = 654321

# Getting influence graph and initial belief from seed:
def get_influence(seed, num_agents, minimum_influence = 0, diagonal_value = 1):
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

def get_rationality(seed, num_agents):
    np.random.seed(seed)
    # return build_rat_graph(Rationality.RANDOM,num_agents)
    # return build_rat_graph(Rationality.CONSTANT,num_agents=num_agents,rationality_value=np.random.uniform(-1,1))
    # return build_rat_graph(Rationality.PER_AGENT,num_agents=num_agents,rationality_values=np.random.uniform(-1,1,num_agents))
    return build_rat_graph(Rationality.PER_AGENT,num_agents=num_agents,rationality_values=np.random.uniform(0,1,num_agents))

#Computing the final state by using the formula, if possible:
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns
def get_weights(influence_graph):
    sist = []
    for i in range(len(influence_graph)):
        for j in range(i+1,len(influence_graph)):
            sist.append([0 for i in range(len(influence_graph))])
            sist[-1][i]+=influence_graph[j][i]
            sist[-1][j]-=influence_graph[i][j]
    return nullspace(sist)
def get_final_state(influence_graph,initial_belief):
    weights = get_weights(influence_graph)
    if weights.shape[0]>0 and weights.shape[1]>0:
        weights = weights[:,0]
        neighbours = [np.count_nonzero(influence_graph[:, i]) for i, _ in enumerate(initial_belief)]
        weights = weights * neighbours
        return np.sum(initial_belief * weights)/np.sum(weights)
    else:
        return []


np.random.seed(GLOBAL_SEED)
def test_final_results_equal(
    container, 
    keys, 
    test_name, 
    number_of_sims = 100000, 
    nagents = 100,
    minimum_influence = 0,
    diagonal_value = 1
    ):
    if not os.path.isdir(f"./generated/{GLOBAL_SEED}_equals"):
        os.mkdir(f"./generated/{GLOBAL_SEED}_equals")
    
    # Seeds for 
    seedsUsed = [(
        np.random.randint(0, 2**32-1),
        np.random.randint(0, 2**32-1),
        np.random.randint(0, 2**32-1)
        ) for i in range(number_of_sims)]

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
        rand_rat = get_rationality(seedsUsed[i][2],nagents)

        funs = [container.get_function(key) for key in keys]
        
        final_states = []
        for fun in funs:
            raw = Simulation(rand_bel,rand_inf,rand_rat,fun)
            final_states.append(raw.get_final_state(tolerance = 0))
        
        arq = open(f"./generated/{GLOBAL_SEED}_equals/{test_name}","r+")
        for j in range(len(final_states)):
            out = False
            for l in range(j+1,len(final_states)):
                if np.linalg.norm(final_states[j]-final_states[l]) >  1e-2:
                    print("Difference found!")
                    arq.seek(0,os.SEEK_END)
                    arq.write(f"({seedsUsed[i][0]},{seedsUsed[i][1]},{seedsUsed[i][2]})\n")
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
        blf_seed = tup[0]
        inf_seed = tup[1]
        rat_seed = tup[2]

        blf = get_initial_belief(blf_seed,nagents)
        inf = get_influence(inf_seed,nagents)
        rat = get_rationality(rat_seed,nagents)

        funs = [container.get_function(key) for key in keys]
        completed_sims = []
        for fun in funs:
            raw = Simulation(blf,inf,rat,fun)
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
    diagonal_value = 1
):
    if not os.path.isdir(f"./generated/{GLOBAL_SEED}_value"):
        os.mkdir(f"./generated/{GLOBAL_SEED}_value")
    seedsUsed = [(np.random.randint(0, 2**32-1),np.random.randint(0, 2**32-1),np.random.randint(0, 2**32-1)) for i in range(number_of_sims)]

    initPos = 0
    if os.path.isfile(f"./generated/{GLOBAL_SEED}_value/{test_name}"):
        arq = open(f"./generated/{GLOBAL_SEED}_value/{test_name}", "r")
        initPos = int(arq.readline())
        arq.close()
    else:
        arq = open(f"./generated/{GLOBAL_SEED}_value/{test_name}", "w")
        arq.write("000000\n")
        arq.close()
    correct=0
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
        rand_rat = get_rationality(seedsUsed[i][2])
        funs = [container.get_function(key) for key in keys]
        
        final_states = []
        for fun in funs:
            raw = Simulation(rand_bel,rand_inf,rand_rat,fun)
            final_states.append(raw.get_final_state(tolerance = 0))
        
        predicted = []
        for fun in funs:
            predicted.append(get_final_state(rand_inf,rand_bel))

        arq = open(f"./generated/{GLOBAL_SEED}_value/{test_name}","r+")
        for j in range(len(final_states)):
            if predicted[j] and np.all(abs(final_states[j]-final_states[j][0]) <= 1e-5):
                if np.linalg.norm(final_states[j] - predicted[j])>1e-3:
                    print(f'\nDifference found!: k= {rand_rat},\n result= {final_states[j]},\n predicted = {predicted[j]}\n influence =\n {rand_inf}\n initbel = {rand_bel}')
                    arq.seek(0,os.SEEK_END)
                    arq.write(f"({seedsUsed[i][0]},{seedsUsed[i][1]},{seedsUsed[i][2]})\n")
                    break
                else:
                    correct+=1
        arq.seek(0)
        arq.write(str(initPos).zfill(6))
        arq.close()
    print(f'Correct predictions: {100*correct/(number_of_sims*len(funs))}%')

