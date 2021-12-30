# bgc-timesteps
Marine ecosystem models are important to identify the processes that affects for example the global carbon cycle. Computation of an annual periodic solution (i.e., a steady annual cycle) for these models requires a high computational effort.

To reduce this effort, we apply larger time steps for the spin-up calculation (i.e., a long-time integration for the calcuation of a steady annual cycle). Using larger time steps shortens the runtime of the spin-up obviously. As an application of the use of larger time steps, we implemented two algorithms (a step size control algorithm and a decreasing time step algorithm) that automatically adapt the time steps during the spin-up calculation in order to use the time steps always as large as possible.


## Installation

To clone this project with **git** run:
>git clone https://github.com/slawig/bgc-timesteps.git


## Usage

The project consists of the Python package timesteps and Python scripts to start the simulation using different time steps in the directory TimestepsSimulation.

The Python packages util is available in the repository https://github.com/slawig/bgc-util.git.


### Python package timesteps

This package summarizes the functions to reduce the computational effort of marine ecosystem models using larger time steps.

This package contains three subpackages:
- decreasingTimesteps:
  Containing the functions for the decreasing time steps algorithm that automatically reduces the time steps during the spin-up.
- stepSizeControl:
  Containing the functions for the step size control algorithm that estimates the local discretization error after a given number of model years and adapt accordingly the time step.
- timesteps:
  Containing functions to use larger time steps for the spin-up calculation.



### Python scripts

Python scripts exist for the applications to start the simulations and evaluate them.

The scripts are available in the directory `TimestepsSimulation`. There are four groups of scripts:
* Scripts using the decreasing time steps algorithm:
  The script `DecreasingTimestep.py` starts simulations using the decreasing time steps algorithm. The script `DecreasingTimestep_Plot.py` visualizes the results.
* Scripts using the step size control algorithm:
  The script `StepSizeControl_Plot.py` visualizes the results using the step size control algorithm.
* Scripts using larger time steps:
  The script `Timesteps.py` starts the simulations using larger time steps and the script `Timestep_Plot.py` visualizes the results.
* Script to create the plots using the two algorithms with an automatic time steps adjustment:
  The script `AutomaticTimestepAdjustment_Plot.py` generates from the data provided on [Zenodo](https://doi.org/10.5281/zenodo.5644003) the figures that are shown in the draft of the paper with the title "Automatic time step adjustment for shortening the runtime of the simulation of marine ecosystem models". A description of how to use this script is included in the Wiki.



## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
