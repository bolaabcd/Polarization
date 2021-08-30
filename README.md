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



## The `Update_Functions` Class

### `BF_Update_Functions` Class

## The `Esteban_Ray_polarization` Class

### `Polarization_Measure` Interface

## The `ManySimulations` Class


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
## Alternative Functions for Simulations

## Helper Functions

