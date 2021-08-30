# A Model for the Dynamics of Polarization

Implemented for Python 3.

## Dependencies

This project depends on [NumPy](https://numpy.org/index.html) (tested on 1.19.0) which may be installed from a console using pip:

```
python3 -m pip install numpy
```

Optionally, we recommend installing [Matplotlib](https://matplotlib.org/) for creating graphs and charts, and [Jupyter Notebooks](https://jupyter.org/index.html) for running simulations interactively.

```
python3 -m pip install matplotlib notebook
```

To install all dependencies you can also run:

```
python3 -m pip install -r requirements.txt
```

## Running Simulations
If you want to try, or recreate, the simulations in `ExampleSimulations.ipynb` for yourself, download this repository, install the corresponding dependencies, and run the jupyter notebook server with access to all .py files of the project. For example, in the same folder as the .py files are located, run:

```
jupyter notebook
```

Select `ExampleSimulations.ipynb`, and make sure to set it as trusted, or just re-run the desired cells.

If you want to run many of the possible simulations and save the results to PDF, do the same as above but with `ManySimlations.ipynb` (and make sure to create the 'ags' and the 'pols' folders, because the results will be placed in these folders).


## The `BF_Update_Functions` Class
This class contains all functions that implement the Backfire-Effect. For now you can select only values of k that were pre-created (Should be fixed in a few updates).

### `Update_Functions` Class
All update functions should extend this class. It contains the Confirmation Bias update function, the Classic update function, and methods to add and get new functions.

## The `Esteban_Ray_polarization` Class
This class implements the Esteban-Ray polarization measure.

### `Polarization_Measure` Interface
All polarization measure functions should extend this interface. It doesn't do anything.

## The `ManySimulations` Class
This class can run and plot many simulations at once.

## Initial Belief Configurations
The definition of `Belief`.{`UNIFORM`, `MILD`, `EXTREME`, `TRIPLE`} is as follows:

There is a new function that allows us to generate new initial belief configurations based on a 5 bins Esteban-Ray Polarization measure, evently distributing all agents in clusters between the [0, 1] interval.

| Belief      | [0, 0.2) | [0.2, 0.4) | [0.4, 0.6) | [0.6, 0.8) | [0.8, 1] |
| ----------- | :------: | :--------: | :--------: | :--------: | :------: |
| UNIFORM     | o | o | o | o | o |
| MILD        |   | o |   | o |   |
| EXTREME     | o |   |   |   | o |
| TRIPLE      | o |   | o |   | o |

To generate such configurations `build_belief` is provided.