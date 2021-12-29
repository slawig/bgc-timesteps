#!/usr/bin/env python
# -*- coding: utf8 -*

import os
from system.system import DATA_PATH, PYTHON_PATH, BACKUP_PATH, FIGURE_PATH

PATH = os.path.join(DATA_PATH, 'LatinHypercubeSample')
PROGRAM_PATH = os.path.join(PYTHON_PATH, 'TimestepsSimulation')
PATH_FIGURE = os.path.join(FIGURE_PATH, 'DecreasingTimesteps')

DB_PATH = os.path.join(PATH, 'Database', 'DecreasingTimesteps_Database.db')

DEFAULT_PYTHONPATH = os.path.join(PYTHON_PATH, 'util') + ':' + os.path.join(PYTHON_PATH, 'timesteps')

PARAMETERID_MAX = 100

CONCENTRATIONID_DICT = {'N': 0, 'N-DOP': 1, 'NP-DOP': 2, 'NPZ-DOP': 3, 'NPZD-DOP': 4, 'MITgcm-PO4-DOP': 1}

PATTERN_JOBFILE = 'Jobfile.{:s}.ParameterId_{:0>3d}.Timestep_{:d}dt.ConcentrationId_{:d}.Tolerance_{:.1e}.Year_{:d}.txt'
PATTERN_LOGFILE = 'Logfile.{:s}.ParameterId_{:0>3d}.Timestep_{:d}dt.ConcentrationId_{:d}.Tolerance_{:.1e}.Year_{:d}.log'
PATTERN_JOBOUTPUT = 'Joboutput.{:s}.ParameterId_{:0>3d}.Timestep_{:d}dt.ConcentrationId_{:d}.Tolerance_{:.1e}.Year_{:d}.out'


PATTERN_OUTPUT_FILENAME = 'job_output_{:0>5d}.out'
PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP = '{:0>5d}_{}_output_timestep_{:d}dt.petsc'

#Pattern for figure filenames
PATTERN_FIGURE_SPINUP = 'Spinup.{:s}.ParameterId_{:0>3d}.pdf'
PATTERN_FIGURE_NORM = '{:s}{:s}Norm.{:s}.ParameterId_{:0>3d}.pdf'
PATTERN_FIGURE_SPINUP_NORM = 'ScatterPlot.SpinupNorm_{:s}{:s}.{:s}.pdf'
PATTERN_FIGURE_NORM_REDUCTION = 'ScatterPlot.NormReduction_{:s}{:s}.{:s}.pdf'

#Backup
PATH_BACKUP = os.path.join(BACKUP_PATH, 'LatinHypercubeSample', 'DecreasingTimesteps')
PATTERN_BACKUP_FILENAME = 'DecreasingTimesteps_ParameterId_{:0>5d}.tar.{}'
PATTERN_BACKUP_LOGFILE = 'DecreasingTimesteps_ParameterId_{:0>5d}_Backup_{}_Remove_{}_Restore_{}.log'
COMPRESSION = 'bz2'
COMPRESSLEVEL = 9

