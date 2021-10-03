# Simulations of Multi-Agent Models for polarization in society
This github repository has python code for simulating the evolution of agent opinions in a generalization of the model presented [here](https://link.springer.com/chapter/10.1007/978-3-030-78089-0_2). At the moment, it can represent society as a graph in wich every person is represented as a node, and communication between agents is a directed edge with some "influence" weight associeted to it. Also, we only deal with discrete and syncrhonous update functions, and we model evolution of the opinion on only one proposition, represented as a value between 0 and 1.

Briefly explaining how we represent everything in the model:
* Society graph is represented by it's adjacency matrix, with the influence strength between two agents in the entrances of the matrix if they influence each other. If not, we have 0 in that entrance.
* The current state of society is represented by a belief vector, in wich the *i-th* entrance is the belief value of agent *i* on the proposition **p**.

Below, we will explain in detail what we have implemented in this repository. In the end of the README, we have a brief explanation on how to use the modules in this repository.

## The polarization_measure module
This is a very simple interface for a polarization measure. A polarization measure is basically a function that recieves the vector of beliefs of all agents and returns a real-number value that represents the polarization level of society at a given point in time. To use the measure, you can create an object that implements the interface and call the method `pol_measure`, passing a vector of beliefs of all agents as an argument.

The `Esteban_Ray_polarization.py` file implements the Esteban-Ray measure for polarization.

## The belief_states module
This module contains a function called `build_belief(belief_type,num_agents)`, that recieves the belief type and the number of agents as parameters, and returns a vector of beliefs of size *num_agents*. Note that the *belief_type* is a type of the Enum `Belief`, that is also in this module.

Here we implement five default initial belief states:
* **UNIFORM** is the initial state in wich all agents have their beleifs uniformly distributed between 0 and 1.
* **MILD** is the state that represents a society with opinions split between two middle opinions.
* **EXTREME** is the state that represents a society with two groups of extreme opinions.
* **TRIPLE** is the stat that represents a society with three groups of opinions: two extremes and one in the middle.
* **RANDOM** is simply a random initial state. All belief values are generated with python's `random` module.

## The influence_graphs module
This module has a function called `build_influence`, and its main parameters are `inf_type` (that describes the type of influence graph to build) and `num_agents` (that indicates how many agents should be in the graph). *inf_type* is one of the `Influence` Enum, wich is also in this module.

We implement seven default influence graphs:
* **CLIQUE** represents the case where every agent influences all other agents.
* **GROUP_2_DISCONECTED** represents two completely disconnected groups, that share no communication between then but that if two agents are in the same group, then they have influence over each other.
* **GROUP_2_FAINT** represents two groups with weak influence on each other, but not zero.
* **INFLUENCERS_2_BALANCED** represents a society two distinct influencers, and the resulting graph is balanced.
* **INFLUENCERS_2_UNBALANCED** represents a society with two distinct influencers, and the resulting graph is not balanced.
* **CIRCULAR** builds a circular graph, so one agent influences exactly another and is influenced by one agent.
* **RANDOM** builds a random influence graph. All influence values are generated with python's `random` module.

## The update_functions module
This module implements an interface for update functions (we call anything that extends this class a **function container**), with implementations for the classic and the confirmation bias ones. `get_function` method returns the desired update function, and `add_function` adds one. Note that internally the functions are stored in a dictionary, so we can expand the list of functions as much as we want. 

The Confirmation Bias and Classic functions are stored in the dictionary with elements of the Enum *Update* (that is in this same module) as keys.

We implemented other functions in the `backfire_update_funs` module. They are acessed by using the elements of the *NewUpdate* Enum as keys. The main functions we have are **LINE**, **MODULUS**, **QUADRATIC**, **CUBIC** and **INTERPOLATED**.

## The Simulation module
This module implements the class `Simulation`, that can run the simulation and return the history of polarization and agents beliefs evolution with the method `run(max_time = 100, smart_stop = True`: *max_time* is the time limit of the simulation (this is according to the discrete time steps, not computation time), and *smart_stop* is whether the simulation should stop if it gets the same result in two consecutive time steps.

There is also a method to get the final state to which the system converges if there is such a state: `get_final_state(self,max_time=50000 ,tolerance=1e-6)`, this method basically returns the first belief state in which agents opinions don't change anymore, or the state the system was when it reached the time limit. The *tolerance* parameter specifies how close should two belief states be to be considered the same. 

## The simulation_multiple module
This module implements the class `ManySimulations`, that runs a lot of simulations subsequently. The constructor accepts identifiers of the update functions, initial beliefs, influence graphs, the maximum time of the simulations, if smart_stop should be enabled and the update functions container that should be used.

We now give a brief description of the methods of this class:
* The method `run` runs all simulations and stores the results in the `completed_sims` attribute.
* The method `plot_polarization(saveTo = None)` plots the polarization of all simulations and stores it in `saveTo` (which should be a file or a path to a file), and this parameter is not specified, the resulting plot is simply closed. Because each graph is simply a line, we plot in the same place all graphs that started with the same belief value and has the same influence graph.
* The method `plot_agents(saveTo = None)` does the same as the above one, but with the graphs that represent the evolution of agents, not of polarization. Every distinct update function, influence graph and initial belief state is plotted separately.

## The random_tests module
This module is capable of running many simulations and check if some of the two implemented properties is true, and if it's not true for some test case, we save the seed we used to generate the influence graph, the seed used to generate the initial belief graph, and the value of k. We also implement functions for plotting the cases that didn't match the property analised, and functions recovering the initial belief and the influence graph from their seeds. Below we name and briefly describe these functions:

* **get_influence(seed, num_agents, minimum_influence = 0, diagonal_value = None)** runs the random influence graph generator with the specified seed and parameters and returns it's matrix.
* **get_initial_belief(seed, num_agents)** runs the random initial belief generator with the specified seed and number of agents, and returns it's vector.
* **test_final_results_equal(container, keys, test_name, number_of_sims = 100000, nagents = 100, minimum_influence = 0, diagonal_value = None)** tests if the functions in the *container* identified by the *keys* have the same final value with random values of *k*, random influence graphs and random beliefs. *number_of_sims* specifies how many random simulations to run, *nagents* the number of agents in each simulation, *minimum_influence* and *diagonal_value* are parameters of the random influence graph generator. All cases in which not all functions got the same result are stored in a file called *test_name* inside a folder called 123456_equals (note that 123456 is the global seed, that can be changed manually in the module file).
* **test_final_results_value(container, keys, test_name, number_of_sims = 100000, nagents = 100, minimum_influence = 0, diagonal_value = None)** tests if the final values of the specified functions can be calculated with the formula that seems to work for two agents. All parameters for this function are the same as the ones for the above function.
* **plot_differents(global_seed, test_name, container, keys, nagents = 100, max_time = 1000, saveTo = None, type = "equals")** runs the cases that have seeds stored in the correspondent file and plots them. *saveTo* specifies the path of a file where the plots will be stored, and if not specified, they won't be stored (wich may be useful).

## About the Notebooks
We also have some github notebooks in this repository to help visualize how everything works:

* **ManySimulations** runs simulations for every combination of update function, update graph and initial belief state (ignoring the randomly generated cases), and stores in the folders "pols" and "ags".
* **ExampleSimulations** show some examples of how to use the modules in this repository.

## How to use the modules in this repository

To use the modules in this repository, you simply need to add to the beginning of your python file `import module_name`, replacing "module_name" with the name of the module you want. Notice that some modules depend on other modules, so make sure that you have all the necessary .py files acessible for importing.

Also note that it is necessary to install *numpy* for most of the modules in this repository to work and, if you want to visualize the data, you will also need *matplotlib*. To see the Jupyter Notebooks correctly, you will also need to install Jupyter or use the Google Colab online services.