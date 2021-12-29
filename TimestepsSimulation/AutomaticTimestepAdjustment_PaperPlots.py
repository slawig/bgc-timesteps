#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import matplotlib.pyplot as plt

import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import decreasingTimesteps.constants as DecreasingTimesteps_Constants
from decreasingTimesteps.DecreasingTimestepsPlot import DecreasingTimestepsPlot
import stepSizeControl.constants as StepSizeControl_Constants
from stepSizeControl.StepSizeControlPlot import StepSizeControlPlot
import timesteps.constants as Timesteps_Constants
from timesteps.TimestepsPlot import TimestepsPlot


#Global variables
PATH_FIGURE = '/gxfs_work1/cau/sunip350/metos3d/LatinHypercubeSample/PaperData'

#Path to the DecreasingTimesteps_Database.db and StepControl_Database.db from https://doi.org/10.5281/zenodo.5644003
PATH_DECREASING_TIMESTEPS_DATABASE = '/gxfs_work1/cau/sunip350/metos3d/LatinHypercubeSample/Database/DecreasingTimesteps_Database.db'
PATH_STEPSIZECONTROL_DATABASE = '/gxfs_work1/cau/sunip350/metos3d/LatinHypercubeSample/Database/StepControl_Database.db'

#Path to the Timesteps_Database.db from https://doi.org/10.5281/zenodo.5643706
PATH_TIMESTEPS_DATABASE = '/gxfs_work1/cau/sunip350/metos3d/LatinHypercubeSample/Database/Timesteps_Database.db'



def main(orientation='gmd', fontsize=8):
    """
    Create the plots for the paper

    Parameters
    ----------
    orientation : str, default: 'gmd'
        Orientation of the figure
    fontsize : int, default: 8
        Fontsize used in the figure

    Notes
    -----
        The plots are created using the databases of the decreasing time steps
        algorithm (specified with the global variable
        PATH_DECREASING_TIMESTEPS_DATABASE), using the database of the step
        size control algorithm (specified with the global variable
        PATH_STEPSIZECONTROL_DATABASE) and using the database of different time
        steps (specified with the global variable PATH_TIMESTEPS_DATABASE). The
        plots are saved in the directory defined with the global variable
        PATH_FIGURE.
        Set the four global variables to configure the plots.
    """
    #Create directory for the plots
    os.makedirs(PATH_FIGURE, exist_ok=True)


    #Plot the results using the step size control algorithm
    stepSizeControlPlot = StepSizeControlPlots(orientation=orientation, fontsize=fontsize)

    #Figure 1 (Results using the N model)
    kwargs = {'simulationIds': [(0, True, 'Reference'), (606, False, 'Step size control'), (1362, False, 'Step size control avoiding\nnegative concentrations')]}
    stepSizeControlPlot.plotSpinupData('Fig1', 'N', 0, **kwargs)

    kwargs = {'simulationIds': [(0, '1dt'), (606, 'Step size control'), (1362, 'Step size control avoiding negative concentrations')]}
    stepSizeControlPlot.plotNormData('Fig1', 'N', 0, year=10000, **kwargs)


    #Figure 2 (Norm of differences against relative error)
    stepSizeControlParameter = [(1, 1, 1.0, 1.0, 2.0, 128, 'BoxweightedVol', False, False, False, 'Step size control'), (1, 1, 1.0, 1.0, 2.0, 64, 'BoxweightedVol', True, False, False, 'Incl. avoidance')]
    kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
    kwargs['N'] = {'stepSizeControl': stepSizeControlParameter, 'legend_box': True}
    kwargs['N-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}}
    kwargs['NP-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}}
    kwargs['NPZ-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}}
    kwargs['NPZD-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.99, 'top': 0.9025}}
    kwargs['MITgcm-PO4-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}}

    for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
        stepSizeControlPlot.plotScatterSpinupNorm('Fig2', metos3dModel, **kwargs[metos3dModel])


    #Figure 4 (Saving of computational costs)
    stepSizeControlParameter = [(1, 1, 1.0, 1.0, 2.0, 128, 'BoxweightedVol', False, False, False, 'Step size control'), (1, 1, 1.0, 1.0, 2.0, 64, 'BoxweightedVol', True, False, False, 'Incl. avoidance')]
    kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
    kwargs['N'] = {'stepSizeControl': stepSizeControlParameter, 'legend_box': True}
    kwargs['N-DOP'] = {'stepSizeControl': stepSizeControlParameter}
    kwargs['NP-DOP'] = {'stepSizeControl': stepSizeControlParameter}
    kwargs['NPZ-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.1375, 'bottom': 0.1475, 'right': 0.98, 'top': 0.9025}}
    kwargs['NPZD-DOP'] = {'stepSizeControl': stepSizeControlParameter}
    kwargs['MITgcm-PO4-DOP'] = {'stepSizeControl': stepSizeControlParameter}

    for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
        stepSizeControlPlot.plotScatterNormReduction('Fig4', metos3dModel, **kwargs[metos3dModel])

    stepSizeControlPlot.closeDatabaseConnection()    


    #Plot the results using different constant time steps
    timestepPlot = TimestepsPlots(orientation=orientation, fontsize=fontsize)

    #Figure 3
    kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
    kwargs['N'] = {'subplot_adjust': {'left': 0.145, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}, 'legend_box': True}
    kwargs['N-DOP'] = {'subplot_adjust': {'left': 0.145, 'bottom': 0.17, 'right': 0.9825, 'top': 0.9025}}

    for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
        timestepPlot.plotScatterSpinupNorm('Fig3', metos3dModel, **kwargs[metos3dModel])

    timestepPlot.closeDatabaseConnection()


    #Plot the results using the decreasing time steps algorithm
    decreasingTimestepsPlot = DecreasingTimestepsPlots(orientation=orientation, fontsize=fontsize)

    #Figure 5 (Results for the N model)
    kwargs = {'simulationIds': [(0, 'Reference'), (606, '0.001'), (607, '0.0001')], 'subPlotModelYear': 9000, 'axesResultSmall': [.69, .4, .2, .2]}
    decreasingTimestepsPlot.plotSpinupData('Fig5', 'N', 0, **kwargs)

    kwargs = {'simulationIds': [(0, '1dt'), (606, '0.001'), (607, '0.0001')], 'subPlotModelYear': 9000}
    decreasingTimestepsPlot.plotNormData('Fig5', 'N', 0, norm='2',  year=10000, **kwargs)


    #Figure 6 (Norm of differences against relative error)
    kwargs = {}
    kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
    kwargs['N'] = {'locMarkerBox' : 'upper left', 'toleranceLegend': True, 'legend_box': True, 'xticksminor': [3*10**(-5), 4*10**(-5), 5*10**(-5), 6*10**(-5), 7*10**(-5), 8*10**(-5), 9*10**(-5)], 'xticksminorlabel': [r'$3 \times 10^{-5}$', '', '', r'$6 \times 10^{-5}$', '', '', '']}
    kwargs['N-DOP'] = {'subplot_adjust': {'left': 0.145, 'bottom': 0.17, 'right': 0.9825, 'top': 0.9025}}

    for metos3dModel in  Metos3d_Constants.METOS3D_MODELS:
        decreasingTimestepsPlot.plotScatterSpinupNorm('Fig6', metos3dModel, **kwargs[metos3dModel])


    #Figure 7 (Saving of computational costs)
    kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
    kwargs['N'] = {'locMarkerBox' : 'lower right', 'legend_box': True, 'toleranceLegend': True}

    for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
        decreasingTimestepsPlot.plotScatterNormReduction('Fig7', metos3dModel, **kwargs[metos3dModel])    

    decreasingTimestepsPlot.closeDatabaseConnection()



class StepSizeControlPlots():
    """
    Preparation of plots for the results using step size control

    Attributes
    ----------
    orientation : str
        Orientation of the figure
    fontsize : int
        Fontsize used in the figure
    """

    def __init__(self, orientation='lc2', fontsize=9):
        """
        Constructs the environment to plot the data using step size control
        for the spin up calculation.

        Parameter
        ----------
        orientation : str, default: lc2
            Orientation of the figure
        fontsize : int, default: 9
            Fontsize used in the figure
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize

        self._orientation = orientation
        self._fontsize = fontsize

        self.__stepSizeControlPlot = StepSizeControlPlot(orientation=self._orientation, fontsize=self._fontsize, dbpath=PATH_STEPSIZECONTROL_DATABASE)


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self.__stepSizeControlPlot.closeDatabaseConnection()


    def plotSpinupData(self, directory, metos3dModel, parameterId, ncol=1, subPlot=False, **kwargs):
        """
        Plot the spin up norm.

        Plot the spin up norm for the given biogeochemical model and
        parameterId. The plot includes the spin up norm for the reference
        simulation (using 1dt) and the simulation using the step size control.

        Parameters
        ----------
        directory : str
            Name of the directory for the figure
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            simulationIds : list [tuple], optional
                List for spin up plots using the simulationId and
                label defined in the tuples
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, StepSizeControl_Constants.PARAMETERID_MAX+1)
        assert type(ncol) is int and 0 < ncol
        assert type(subPlot) is bool

        #Parse keyword arguments
        axesResultSmall = kwargs['axesResultSmall'] if 'axesResultSmall' in kwargs and type(kwargs['axesResultSmall']) is list and len(kwargs['axesResultSmall']) == 4 else [.69, .5, .2, .2]
        subPlotModelYear = kwargs['subPlotModelYear'] if 'subPlotModelYear' in kwargs and type(kwargs['subPlotModelYear']) is int and 0 < kwargs['subPlotModelYear'] else 8000
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.1525, 'bottom': 0.165, 'right': 0.9625, 'top': 0.995}
        simulationIds = kwargs['simulationIds'] if 'simulationIds' in kwargs and type(kwargs['simulationIds'] is list) else []
        kwargsLegend = {'labelspacing': 0.3, 'handlelength': 1.0, 'handletextpad': 0.4, 'columnspacing': 1.0, 'borderaxespad': 0.3, 'borderpad': 0.2}

        #Create spin up norm plot
        self.__stepSizeControlPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__stepSizeControlPlot.plot_spinup_data(ncol=ncol, simulationIds=simulationIds, subPlot=subPlot, axesResultSmall=axesResultSmall, subPlotModelYear=subPlotModelYear, **kwargsLegend)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__stepSizeControlPlot.set_legend_box(ncol=ncol)

        self.__stepSizeControlPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        os.makedirs(os.path.join(PATH_FIGURE, directory), exist_ok=True)
        filenameSpinup = os.path.join(PATH_FIGURE, directory, StepSizeControl_Constants.PATTERN_FIGURE_SPINUP.format(metos3dModel, parameterId))
        self.__stepSizeControlPlot.savefig(filenameSpinup)
        self.__stepSizeControlPlot.close_fig()


    def plotNormData(self, directory, model, parameterId, norm='2', trajectory='', year=None, ncol=3, **kwargs):
        """
        Plot the relative error

        Plot the relative error of the spin up for the given biogeochemical
        model and parameterId. The plot includes the relative error for 
        different time steps. The tracer concentrations calculated with the
        spin up using the time step 1dt are used as reference solution.

        Parameters
        ----------
        directory : str
            Name of the directory for the figure
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
        model : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        year : int or None, default: None
            Model year used for the reference solution
            If None, use the same year for the reference solution,
            respectively
        ncol : int, default: 3
            Number of columns for the legend
        **kwargs : dict
            Additional keyword arguments with keys:

            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            simulationIds : list [tuple [int]], optional
                List for spin up plots using the simulationId and
                timestep defined in the tuples
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, StepSizeControl_Constants.PARAMETERID_MAX+1)
        assert trajectory in ['', 'trajectory']
        assert norm in DB_Constants.NORM
        assert year is None or type(year) is int and 0 <= year
        assert type(ncol) is int and 0 < ncol

        #Parse keyword arguments
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.145, 'bottom': 0.165, 'right': 0.9625, 'top': 0.995}
        simulationIds = kwargs['simulationIds'] if 'simulationIds' in kwargs and type(kwargs['simulationIds'] is list) else []
        kwargsLegend = {'handlelength': 1.0, 'handletextpad': 0.4, 'columnspacing': 1.0, 'borderaxespad': 0.3}

        #Create scatter plot
        self.__stepSizeControlPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__stepSizeControlPlot.plot_tracer_norm_data(simulationIds[0][0], simulationIds[1:], ncol=ncol, year=year, **kwargsLegend)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__stepSizeControlPlot.set_legend_box(ncol=ncol)

        self.__stepSizeControlPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])
        os.makedirs(os.path.join(PATH_FIGURE, directory), exist_ok=True)
        filenameNorm = os.path.join(PATH_FIGURE, directory, StepSizeControl_Constants.PATTERN_FIGURE_NORM.format(trajectory, norm, model, parameterId))
        self.__stepSizeControlPlot.savefig(filenameNorm)
        self.__stepSizeControlPlot.close_fig()


    def plotScatterSpinupNorm(self, directory, model, year=10000, norm='2', trajectory='', **kwargs):
        """
        Plot the spin up norm against the norm

        Plot the spin up norm value against the norm value for the given model
        year using the different yearIntervals and tolerances

        Parameters
        ----------
        directory : str
            Name of the directory for the figure
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
        model : str
            Name of the biogeochemical model
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        **kwargs : dict
            Additional keyword arguments with keys:

            legend_box : bool
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            ncol : int, default: 3
                Number of columns for the legend
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        #Parse keyword arguments
        stepSizeControlParameter = kwargs['stepSizeControl'] if 'stepSizeControl' in kwargs and type(kwargs['stepSizeControl']) is list else [(1, 1, 1.0, 1.0, 2.0, 128, 'BoxweightedVol', False, False, False, 'Step size control')]
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.145, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}
        ncol = kwargs['ncol'] if 'ncol' in kwargs and type(kwargs['ncol']) is int and kwargs['ncol'] > 0 else 3
        handlelength = kwargs['handlelength'] if 'handlelength' in kwargs and type(kwargs['handlelength']) is float and 0.0 < kwargs['handlelength'] else 0.5
        handletextpad = kwargs['handlepadtext'] if 'handlepadtext' in kwargs and type(kwargs['handlepadtext']) is float and 0.0 < kwargs['handlepadtext'] else 0.4
        kwarg = {'xticksminor': kwargs['xticksminor'], 'xticksminorlabel': kwargs['xticksminorlabel']} if 'xticksminor' in kwargs and type(kwargs['xticksminor']) is list and 'xticksminorlabel' in kwargs and type(kwargs['xticksminorlabel']) is list else {}

        #Create scatter plot
        self.__stepSizeControlPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__stepSizeControlPlot.plot_scatter_spinup_norm(model, stepSizeControlParameter=stepSizeControlParameter, year=year, norm=norm, trajectory=trajectory, **kwarg)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__stepSizeControlPlot.set_legend_box(ncol=ncol, handlelength=handlelength, handletextpad=handletextpad)

        self.__stepSizeControlPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        os.makedirs(os.path.join(PATH_FIGURE, directory), exist_ok=True)
        filename = os.path.join(PATH_FIGURE, directory, StepSizeControl_Constants.PATTERN_FIGURE_SPINUP_NORM.format(trajectory, norm, model))
        self.__stepSizeControlPlot.savefig(filename)
        self.__stepSizeControlPlot.close_fig()


    def plotScatterNormReduction(self, directory, model, year=10000, norm='2', trajectory='', **kwargs):
        """
        Plot the relative error (norm) against the reduction

        Plot the norm norm value against the reduction of the computational
        costs for the given model year using the different yearIntervals and
        tolerances

        Parameters
        ----------
        directory : str
            Name of the directory for the figure
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
        model : str
            Name of the biogeochemical model
       year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        **kwargs : dict
            Additional keyword arguments with keys:

            legend_box : bool
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            ncol : int, default: 3
                Number of columns for the legend

        NOTES
        -----
        The figure is saved in the directory defined in
        decreasingTimesteps.constants.PATH_FIGURE.
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        #Parse keyword arguments
        stepSizeControlParameter = kwargs['stepSizeControl'] if 'stepSizeControl' in kwargs and type(kwargs['stepSizeControl']) is list else [(1, 1, 1.0, 1.0, 2.0, 128, 'BoxweightedVol', False, False, False, 'Step size control')]
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.1375, 'bottom': 0.1475, 'right': 0.995, 'top': 0.9025}
        ncol = kwargs['ncol'] if 'ncol' in kwargs and type(kwargs['ncol']) is int and kwargs['ncol'] > 0 else 3
        handlelength = kwargs['handlelength'] if 'handlelength' in kwargs and type(kwargs['handlelength']) is float and 0.0 < kwargs['handlelength'] else 0.5
        handletextpad = kwargs['handlepadtext'] if 'handlepadtext' in kwargs and type(kwargs['handlepadtext']) is float and 0.0 < kwargs['handlepadtext'] else 0.4

        #Create scatter plot
        self.__stepSizeControlPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__stepSizeControlPlot.plot_scatter_norm_costreduction(model, stepSizeControlParameter=stepSizeControlParameter, year=year, norm=norm, trajectory=trajectory)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__stepSizeControlPlot.set_legend_box(ncol=ncol, handlelength=handlelength, handletextpad=handletextpad)

        self.__stepSizeControlPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        os.makedirs(os.path.join(PATH_FIGURE, directory), exist_ok=True)
        filename = os.path.join(PATH_FIGURE, directory, StepSizeControl_Constants.PATTERN_FIGURE_NORM_REDUCTION.format(trajectory, norm, model))
        self.__stepSizeControlPlot.savefig(filename)
        self.__stepSizeControlPlot.close_fig()



class DecreasingTimestepsPlots():
    """
    Preparation of plots for the results using decreasing time steps

    Attributes
    ----------
    orientation : str
        Orientation of the figure
    fontsize : int
        Fontsize used in the figure
    """

    def __init__(self, orientation='lc2', fontsize=9):
        """
        Constructs the environment to plot the data using decreasing time steps
        for the spin up calculation.

        Parameter
        ----------
        orientation : str, default: gmd
            Orientation of the figure
        fontsize : int, default: 8
            Fontsize used in the figure
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize

        self._orientation = orientation
        self._fontsize = fontsize

        self.__decreasingTimestepPlot = DecreasingTimestepsPlot(orientation=self._orientation, fontsize=self._fontsize, dbpath=PATH_DECREASING_TIMESTEPS_DATABASE)


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self.__decreasingTimestepPlot.closeDatabaseConnection()


    def plotSpinupData(self, directory,  model, parameterId, ncol=3, subPlot=True, **kwargs):
        """
        Plot the spin up norm.

        Plot the spin up norm for the given biogeochemical model and
        parameterId. The plot includes the spin up norm for the reference
        simulation (using 1dt) and the simulation using the decreasing time
        steps.

        Parameters
        ----------
        directory : str
            Name of the directory for the figure
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
        model : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            yearIntervalList : list [int], optional
                Representation of the spin up norm for each specified
                yearInterval
            yearToleranceList : list [float], optional
                Representation for each specified tolerance
            timestep : int, optional
                Representation for the specified initial time step
            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            simulationIds : list [tuple], optional
                List for spin up plots using the simulationId and
                label defined in the tuples
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, DecreasingTimesteps_Constants.PARAMETERID_MAX+1)
        assert type(ncol) is int and 0 < ncol
        assert type(subPlot) is bool

        #Parse keyword arguments
        yearIntervalList = kwargs['yearIntervalList'] if 'yearIntervalList' in kwargs and type(kwargs['yearIntervalList']) is list else [50, 100, 500]
        toleranceList = kwargs['toleranceList'] if 'toleranceList' in kwargs and type(kwargs['toleranceList']) is list else [0.001, 0.0001]
        timestep = kwargs['timestep'] if 'timestep' in kwargs and kwargs['timestep'] in Metos3d_Constants.METOS3D_TIMESTEPS else Metos3d_Constants.METOS3D_TIMESTEPS[-1]
        axesResultSmall = kwargs['axesResultSmall'] if 'axesResultSmall' in kwargs and type(kwargs['axesResultSmall']) is list and len(kwargs['axesResultSmall']) == 4 else [.69, .5, .2, .2]
        subPlotModelYear = kwargs['subPlotModelYear'] if 'subPlotModelYear' in kwargs and type(kwargs['subPlotModelYear']) is int and 0 < kwargs['subPlotModelYear'] else 8000
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.1525, 'bottom': 0.165, 'right': 0.9625, 'top': 0.995}
        simulationIds = kwargs['simulationIds'] if 'simulationIds' in kwargs and type(kwargs['simulationIds'] is list) else []
        kwargsLegend = {'handlelength': 1.0, 'handletextpad': 0.4, 'columnspacing': 1.0, 'borderaxespad': 0.3}

        #Create scatter plot
        self.__decreasingTimestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__decreasingTimestepPlot.plot_spinup_data(ncol=ncol, simulationIds=simulationIds, subPlot=subPlot, axesResultSmall=axesResultSmall, subPlotModelYear=subPlotModelYear, **kwargsLegend)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__decreasingTimestepPlot.set_legend_box(ncol=ncol)

        self.__decreasingTimestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        os.makedirs(os.path.join(PATH_FIGURE, directory), exist_ok=True)
        filenameSpinup = os.path.join(PATH_FIGURE, directory, DecreasingTimesteps_Constants.PATTERN_FIGURE_SPINUP.format(model, parameterId))

        self.__decreasingTimestepPlot.savefig(filenameSpinup)
        self.__decreasingTimestepPlot.close_fig()


    def plotNormData(self, directory, model, parameterId, norm='2', trajectory='', year=None, ncol=3, **kwargs):
        """
        Plot the relative error

        Plot the relative error of the spin up for the given biogeochemical
        model and parameterId. The plot includes the relative error for 
        different time steps. The tracer concentrations calculated with the
        spin up using the time step 1dt are used as reference solution.

        Parameters
        ----------
        directory : str
            Name of the directory for the figure
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
        model : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        year : int or None, default: None
            Model year used for the reference solution
            If None, use the same year for the reference solution,
            respectively
        ncol : int, default: 3
            Number of columns for the legend
        **kwargs : dict
            Additional keyword arguments with keys:

            timestepList : list [int], optional
                Representation of the spin up norm for each specified timestep
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            simulationIds : list [tuple [int]], optional
                List for spin up plots using the simulationId and
                timestep defined in the tuples
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, DecreasingTimesteps_Constants.PARAMETERID_MAX+1)
        assert trajectory in ['', 'trajectory']
        assert norm in DB_Constants.NORM
        assert year is None or type(year) is int and 0 <= year
        assert type(ncol) is int and 0 < ncol

        #Parse keyword arguments
        yearIntervalList = kwargs['yearIntervalList'] if 'yearIntervalList' in kwargs and type(kwargs['yearIntervalList']) is list else [50, 100, 500]
        toleranceList = kwargs['toleranceList'] if 'toleranceList' in kwargs and type(kwargs['toleranceList']) is list else [0.001, 0.0001]
        timestep = kwargs['timestep'] if 'timestep' in kwargs and kwargs['timestep'] in Metos3d_Constants.METOS3D_TIMESTEPS else Metos3d_Constants.METOS3D_TIMESTEPS[-1]
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.145, 'bottom': 0.165, 'right': 0.9625, 'top': 0.995}
        simulationIds = kwargs['simulationIds'] if 'simulationIds' in kwargs and type(kwargs['simulationIds'] is list) else []
        kwargsLegend = {'handlelength': 1.0, 'handletextpad': 0.4, 'columnspacing': 1.0, 'borderaxespad': 0.3}


        #Create scatter plot
        self.__decreasingTimestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__decreasingTimestepPlot.plot_tracer_norm_data(simulationIds[0][0], simulationIds[1:], ncol=ncol, year=year, **kwargsLegend)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__decreasingTimestepPlot.set_legend_box(ncol=ncol)

        self.__decreasingTimestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        os.makedirs(os.path.join(PATH_FIGURE, directory), exist_ok=True)
        filenameNorm = os.path.join(PATH_FIGURE, directory, DecreasingTimesteps_Constants.PATTERN_FIGURE_NORM.format(trajectory, norm, model, parameterId))
        self.__decreasingTimestepPlot.savefig(filenameNorm)
        self.__decreasingTimestepPlot.close_fig()


    def plotScatterSpinupNorm(self, directory,  model, year=10000, norm='2', trajectory='', **kwargs):
        """
        Plot the spin up norm against the norm

        Plot the spin up norm value against the norm value for the given model
        year using the different yearIntervals and tolerances

        Parameters
        ----------
        directory : str
            Name of the directory for the figure
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot

        model : str
            Name of the biogeochemical model
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        **kwargs : dict
            Additional keyword arguments with keys:

            yearIntervalList : list [int]
                Representation of the spin up norm for each specified
                yearInterval
            yearToleranceList : list [float]
                Representation for each specified tolerance
            timestep : int
                Representation for the specified initial time step
            legend_box : bool
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            ncol : int, default: 3
                Number of columns for the legend
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        #Parse keyword arguments
        yearIntervalList = kwargs['yearIntervalList'] if 'yearIntervalList' in kwargs and type(kwargs['yearIntervalList']) is list else [50, 100, 500]
        toleranceList = kwargs['toleranceList'] if 'toleranceList' in kwargs and type(kwargs['toleranceList']) is list else [0.001, 0.0001]
        timestep = kwargs['timestep'] if 'timestep' in kwargs and kwargs['timestep'] in Metos3d_Constants.METOS3D_TIMESTEPS else Metos3d_Constants.METOS3D_TIMESTEPS[-1]
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.145, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}
        ncol = kwargs['ncol'] if 'ncol' in kwargs and type(kwargs['ncol']) is int and kwargs['ncol'] > 0 else 3
        handlelength = kwargs['handlelength'] if 'handlelength' in kwargs and type(kwargs['handlelength']) is float and 0.0 < kwargs['handlelength'] else 0.5
        handletextpad = kwargs['handlepadtext'] if 'handlepadtext' in kwargs and type(kwargs['handlepadtext']) is float and 0.0 < kwargs['handlepadtext'] else 0.4
        locMarkerBox = kwargs['locMarkerBox'] if 'locMarkerBox' in kwargs and type(kwargs['locMarkerBox']) is str else 'lower right'
        kwarg = {'xticksminor': kwargs['xticksminor'], 'xticksminorlabel': kwargs['xticksminorlabel']} if 'xticksminor' in kwargs and type(kwargs['xticksminor']) is list and 'xticksminorlabel' in kwargs and type(kwargs['xticksminorlabel']) is list else {}
        toleranceLegend = kwargs['toleranceLegend'] if 'toleranceLegend' in kwargs and type(kwargs['toleranceLegend']) is bool else False

        #Create scatter plot
        self.__decreasingTimestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__decreasingTimestepPlot.plot_scatter_spinup_norm(model, timestep=timestep, yearIntervalList=yearIntervalList, toleranceList=toleranceList, year=year, norm=norm, trajectory=trajectory, locMarkerBox=locMarkerBox, toleranceLegend=toleranceLegend, **kwarg)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__decreasingTimestepPlot.set_legend_box(ncol=ncol, handlelength=handlelength, handletextpad=handletextpad)

        self.__decreasingTimestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        os.makedirs(os.path.join(PATH_FIGURE, directory), exist_ok=True)
        filename = os.path.join(PATH_FIGURE, directory, DecreasingTimesteps_Constants.PATTERN_FIGURE_SPINUP_NORM.format(trajectory, norm, model))
        self.__decreasingTimestepPlot.savefig(filename)
        self.__decreasingTimestepPlot.close_fig()


    def plotScatterNormReduction(self, directory, model, year=10000, norm='2', trajectory='', **kwargs):
        """
        Plot the relative error (norm) against the reduction

        Plot the norm norm value against the reduction of the computational
        costs for the given model year using the different yearIntervals and
        tolerances

        Parameters
        ----------
        directory : str
            Name of the directory for the figure
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
        model : str
            Name of the biogeochemical model
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        **kwargs : dict
            Additional keyword arguments with keys:

            yearIntervalList : list [int]
                Representation of the spin up norm for each specified
                yearInterval
            yearToleranceList : list [float]
                Representation for each specified tolerance
            timestep : int
                Representation for the specified initial time step
            legend_box : bool
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            ncol : int, default: 3
                Number of columns for the legend
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        #Parse keyword arguments
        yearIntervalList = kwargs['yearIntervalList'] if 'yearIntervalList' in kwargs and type(kwargs['yearIntervalList']) is list else [50, 100, 500]
        toleranceList = kwargs['toleranceList'] if 'toleranceList' in kwargs and type(kwargs['toleranceList']) is list else [0.001, 0.0001]
        timestep = kwargs['timestep'] if 'timestep' in kwargs and kwargs['timestep'] in Metos3d_Constants.METOS3D_TIMESTEPS else Metos3d_Constants.METOS3D_TIMESTEPS[-1]
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.125, 'bottom': 0.1475, 'right': 0.995, 'top': 0.9025}
        ncol = kwargs['ncol'] if 'ncol' in kwargs and type(kwargs['ncol']) is int and kwargs['ncol'] > 0 else 3
        handlelength = kwargs['handlelength'] if 'handlelength' in kwargs and type(kwargs['handlelength']) is float and 0.0 < kwargs['handlelength'] else 0.5
        handletextpad = kwargs['handlepadtext'] if 'handlepadtext' in kwargs and type(kwargs['handlepadtext']) is float and 0.0 < kwargs['handlepadtext'] else 0.4
        locMarkerBox = kwargs['locMarkerBox'] if 'locMarkerBox' in kwargs and type(kwargs['locMarkerBox']) is str else 'lower right'
        toleranceLegend = kwargs['toleranceLegend'] if 'toleranceLegend' in kwargs and type(kwargs['toleranceLegend']) is bool else False

        #Create scatter plot
        self.__decreasingTimestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__decreasingTimestepPlot.plot_scatter_norm_costreduction(model, timestep=timestep, yearIntervalList=yearIntervalList, toleranceList=toleranceList, year=year, norm=norm, trajectory=trajectory, locMarkerBox=locMarkerBox, toleranceLegend=toleranceLegend)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__decreasingTimestepPlot.set_legend_box(ncol=ncol, handlelength=handlelength, handletextpad=handletextpad)

        self.__decreasingTimestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        os.makedirs(os.path.join(PATH_FIGURE, directory), exist_ok=True)
        filename = os.path.join(PATH_FIGURE, directory, DecreasingTimesteps_Constants.PATTERN_FIGURE_NORM_REDUCTION.format(trajectory, norm, model))
        self.__decreasingTimestepPlot.savefig(filename)
        self.__decreasingTimestepPlot.close_fig()



class TimestepsPlots():
    """
    Preparation of plots for the results using different time steps

    Preparation of plots for the results using different time steps for the
    spin up calculation.

    Attributes
    ----------
    orientation : str
        Orientation of the figure
    fontsize : int
        Fontsize used in the figure
    """

    def __init__(self, orientation='gmd', fontsize=8):
        """
        Constructs the environment to plot the data using different time steps
        for the spin up calculation.

        Parameter
        ----------
        orientation : str, default: gmd
            Orientation of the figure
        fontsize : int, default: 8
            Fontsize used in the figure
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize

        self._orientation = orientation
        self._fontsize = fontsize

        self.__timestepPlot = TimestepsPlot(orientation=self._orientation, fontsize=self._fontsize)


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self.__timestepPlot.closeDatabaseConnection()


    def plotScatterSpinupNorm(self, directory, model, year=10000, norm='2', trajectory='', **kwargs):
        """
        Plot the spin up norm against the norm

        Plot the spin up norm value against the norm value for the given model
        year using the different time steps

        Parameters
        ----------
        directory : str
            Name of the directory for the figure
        model : str
            Name of the biogeochemical model
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        **kwargs : dict
            Additional keyword arguments with keys:

            timestepList : list [int]
                Representation of the spin up norm for each specified timestep
            legend_box : bool
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            ncol : int, default: 3
                Number of columns for the legend

        NOTES
        -----
        The figure is saved in the directory defined in
        timesteps.constants.PATH_FIGURE.
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        #Parse keyword arguments
        timestepList = kwargs['timestepList'] if 'timestepList' in kwargs and type(kwargs['timestepList']) is list else Metos3d_Constants.METOS3D_TIMESTEPS[1:]
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.145, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}
        ncol = kwargs['ncol'] if 'ncol' in kwargs and type(kwargs['ncol']) is int and kwargs['ncol'] > 0 else 6
        handlelength = kwargs['handlelength'] if 'handlelength' in kwargs and type(kwargs['handlelength']) is float and 0.0 < kwargs['handlelength'] else 0.5
        handletextpad = kwargs['handlepadtext'] if 'handlepadtext' in kwargs and type(kwargs['handlepadtext']) is float and 0.0 < kwargs['handlepadtext'] else 0.4
        oscillationLegend = kwargs['oscillationLegend'] if 'oscillationLegend' in kwargs and type(kwargs['oscillationLegend']) is bool else False
        oscillation = kwargs['oscillation'] if 'oscillation' in kwargs and type(kwargs['oscillation']) is bool else False

        self.__timestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__timestepPlot.plot_scatter_spinup_norm(model, year=year, norm=norm, trajectory=trajectory, timestepList=timestepList, oscillationIdentification=oscillation, oscillationLegend=oscillationLegend)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__timestepPlot.set_legend_box(ncol=ncol, handlelength=handlelength, handletextpad=handletextpad)

        self.__timestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        os.makedirs(os.path.join(PATH_FIGURE, directory), exist_ok=True)
        filename = os.path.join(PATH_FIGURE, directory, Timesteps_Constants.PATTERN_FIGURE_SPINUP_NORM.format(trajectory, norm, model))
        self.__timestepPlot.savefig(filename)
        self.__timestepPlot.close_fig()


if __name__ == '__main__':
    main()

