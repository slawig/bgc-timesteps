#!/usr/bin/env python
# -*- coding: utf8 -*

import itertools
import numpy as np
import os
import matplotlib.pyplot as plt

import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import stepSizeControl.constants as StepSizeControl_Constants
from stepSizeControl.StepSizeControlPlot import StepSizeControlPlot


def main(orientation='gmd', fontsize=8, plotSpinup=False, plotNorm=False, plotRelationSpinupNorm=True, plotRelationNormReduction=True):
    """
    Plot the results using step size control for the spin up

    Create plots of the results using step size control for the spin up
    calculation.

    Parameters
    ----------
    orientation : str
        Orientation of the figure
    fontsize : int
        Fontsize used in the figure
    plotSpinup : bool, default: False
        If True, plot the figure of the spin up norm
    plotNorm : bool, default: False
        If True, plot the figures for the norm
    plotRelationSpinupNorm : bool, default: False
        If True, plot the figures of the relation between spin up norm and
        relative error
    plotRelationNormReduction : bool, default: False
        If True, plot the figures of the relation between relative error and
        the reduction of the computational costs
    """
    assert type(orientation) is str
    assert type(fontsize) is int and 0 < fontsize
    assert type(plotSpinup) is bool
    assert type(plotNorm) is bool
    assert type(plotRelationSpinupNorm) is bool
    assert type(plotRelationNormReduction) is bool

    stepSizeControlPlot = StepSizeControlPlots(orientation=orientation, fontsize=fontsize)

    #Spin up norm plot
    if plotSpinup:
        parameterId = 0
        kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
        kwargs['N'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(0, True, 'Reference'), (6, False, 'Step size control'), (1386, False, 'Step size control avoiding\nnegative concentrations')]}
        kwargs['N-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(1, True, 'Reference'), (16, False, 'Step size control'), (1387, False, 'Step size control avoiding\nnegative concentrations')]}
        kwargs['NP-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(2, True, 'Reference'), (26, False, 'Step size control'), (1388, False, 'Step size control avoiding\nnegative concentrations')]}
        kwargs['NPZ-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(3, True, 'Reference'), (36, False, 'Step size control'), (1389, False, 'Step size control avoiding\nnegative concentrations')]}
        kwargs['NPZD-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(4, True, 'Reference'), (46, False, 'Step size control'), (1390, False, 'Step size control avoiding\nnegative concentrations')]}
        kwargs['MITgcm-PO4-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(5, True, 'Reference'), (56, False, 'Step size control'), (1391, False, 'Step size control avoiding\nnegative concentrations')]}

        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            stepSizeControlPlot.plotSpinupData(metos3dModel, parameterId, **kwargs[metos3dModel])


    #Norm data plot
    if plotNorm:
        year = 10000
        normList = ['2']
        parameterIdList = [0]
        metos3dModelList = Metos3d_Constants.METOS3D_MODELS

        kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
        kwargs['N'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(0, '1dt'), (6, 'Step size control'), (1386, 'Step size control avoiding negative concentrations')]}
        kwargs['N-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(1, '1dt'), (16, 'Step size control'), (1387, 'Step size control avoiding negative concentrations')]}
        kwargs['NP-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(2, '1dt'), (26, 'Step size control'), (1388, 'Step size control avoiding negative concentrations')]}
        kwargs['NPZ-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(3, '1dt'), (36, 'Step size control'), (1389, 'Step size control avoiding negative concentrations')]}
        kwargs['NPZD-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(4, '1dt'), (46, 'Step size control'), (1390, 'Step size control avoiding negative concentrations')]}
        kwargs['MITgcm-PO4-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(5, '1dt'), (56, 'Step size control'), (1391, 'Step size control avoiding negative concentrations')]}

        for (norm, parameterId, metos3dModel) in list(itertools.product(normList, parameterIdList, metos3dModelList)):
            stepSizeControlPlot.plotNormData(metos3dModel, parameterId, norm=norm, year=year, **kwargs[metos3dModel])


    #Plot relation between spin up norm and relative error for all parameter vectors
    if plotRelationSpinupNorm:
        year = 10000
        stepSizeControlParameter = [(1, 1, 1.0, 1.0, 2.0, 128, 'BoxweightedVol', False, False, False, 'Step size control'), (1, 1, 1.0, 1.0, 2.0, 64, 'BoxweightedVol', True, False, False, 'Incl. avoidance')]
        kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
        kwargs['N'] = {'stepSizeControl': stepSizeControlParameter, 'legend_box': True}
        kwargs['N-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}}
        kwargs['NP-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}}
        kwargs['NPZ-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}}
        kwargs['NPZD-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.99, 'top': 0.9025}}
        kwargs['MITgcm-PO4-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.16, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}}

        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            stepSizeControlPlot.plotScatterSpinupNorm(metos3dModel, year=year, **kwargs[metos3dModel])


    #Plot relation between relative error and the reduction of computational costs
    if plotRelationNormReduction:
        year = 10000
        stepSizeControlParameter = [(1, 1, 1.0, 1.0, 2.0, 128, 'BoxweightedVol', False, False, False, 'Step size control'), (1, 1, 1.0, 1.0, 2.0, 64, 'BoxweightedVol', True, False, False, 'Incl. avoidance')]
        kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
        kwargs['N'] = {'stepSizeControl': stepSizeControlParameter, 'legend_box': True}
        kwargs['N-DOP'] = {'stepSizeControl': stepSizeControlParameter}
        kwargs['NP-DOP'] = {'stepSizeControl': stepSizeControlParameter}
        kwargs['NPZ-DOP'] = {'stepSizeControl': stepSizeControlParameter, 'subplot_adjust': {'left': 0.1375, 'bottom': 0.1475, 'right': 0.98, 'top': 0.9025}}
        kwargs['NPZD-DOP'] = {'stepSizeControl': stepSizeControlParameter}
        kwargs['MITgcm-PO4-DOP'] = {'stepSizeControl': stepSizeControlParameter}

        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            stepSizeControlPlot.plotScatterNormReduction(metos3dModel, year=year, **kwargs[metos3dModel])

    stepSizeControlPlot.closeDatabaseConnection()



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

        self.__stepSizeControlPlot = StepSizeControlPlot(orientation=self._orientation, fontsize=self._fontsize)


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self.__stepSizeControlPlot.closeDatabaseConnection()


    def plotSpinupData(self, metos3dModel, parameterId, ncol=1, subPlot=False, **kwargs):
        """
        Plot the spin up norm.

        Plot the spin up norm for the given biogeochemical model and
        parameterId. The plot includes the spin up norm for the reference
        simulation (using 1dt) and the simulation using the step size control.

        Parameters
        ----------
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

            stepYearList : list [int], optional
                Representation of the spin up norm for each specified stepYear
                used for the step size control
            toleranceList : list [float], optinal
                Representation of the spin up norm for each specified tolerance
                used for the step size control
            rhoList : list [float], optional
                Representation of the spin up norm for each specified parameter
                value of rho used for the step size control
            etaList : list [float], optional
                Representation of the spin up norm for each specified parameter
                value of eta used for the step size control
            normList : list [str], optional
                Representation of the spin up norm for each specified norm used
                for the step size control
            startTimestepList : list [int], optional
                Representation of the spin up norm for each specified initial
                time step used for the step size control
            axesResultSmall : list [float], optional
                Dimensions of the subplot
            subplot_adjust : dict [str, float], optional
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool, optional
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            additionalSimulationIds : list [tuple], optional
                List for additional spin up plots using the simulationId and
                label defined in the tuples
            onlyAdditionalSimulationIds : bool, optional
                If True, plot only the spin up of the simulations defined by
                the simulationIds in the additionalSimulationIds list
            filenamePrefix : str or None, optional
                Prefix of the filename for the figure

        NOTES
        -----
        The figure is saved in the directory defined in
        stepSizeControl.constants.PATH_FIGURE.
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, StepSizeControl_Constants.PARAMETERID_MAX+1)
        assert type(ncol) is int and 0 < ncol
        assert type(subPlot) is bool

        #Parse keyword arguments
        axesResultSmall = kwargs['axesResultSmall'] if 'axesResultSmall' in kwargs and type(kwargs['axesResultSmall']) is list and len(kwargs['axesResultSmall']) == 4 else [.69, .5, .2, .2]
        subPlotModelYear = kwargs['subPlotModelYear'] if 'subPlotModelYear' in kwargs and type(kwargs['subPlotModelYear']) is int and 0 < kwargs['subPlotModelYear'] else 8000
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.1525, 'bottom': 0.165, 'right': 0.9625, 'top': 0.995}
        onlyAdditionalSimulationIds = kwargs['onlyAdditionalSimulationIds'] if 'onlyAdditionalSimulationIds' in kwargs else False
        additionalSimulationIds = kwargs['additionalSimulationIds'] if 'additionalSimulationIds' in kwargs and type(kwargs['additionalSimulationIds'] is list)else []
        filenameSpinup = os.path.join(StepSizeControl_Constants.PATH_FIGURE, 'Spinup', metos3dModel, kwargs['filenamePrefix'] + StepSizeControl_Constants.PATTERN_FIGURE_SPINUP.format(metos3dModel, parameterId) if 'filenamePrefix' in kwargs and type(kwargs['filenamePrefix']) is str else StepSizeControl_Constants.PATTERN_FIGURE_SPINUP.format(metos3dModel, parameterId))
        kwargsLegend = {'labelspacing': 0.3, 'handlelength': 1.0, 'handletextpad': 0.4, 'columnspacing': 1.0, 'borderaxespad': 0.3, 'borderpad': 0.2}

        #Create spin up norm plot
        self.__stepSizeControlPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        if onlyAdditionalSimulationIds:
            self.__stepSizeControlPlot.plot_spinup_data(ncol=ncol, simulationIds=additionalSimulationIds, subPlot=subPlot, axesResultSmall=axesResultSmall, subPlotModelYear=subPlotModelYear, **kwargsLegend)
        else:
            #TODO Setzen der notwednigen simulationIds und labels (abhaengig vom Modell und ParameterId)
            self.__stepSizeControlPlot.plot_spinup_data(ncol=ncol, simulationIds=additionalSimulationIds)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__stepSizeControlPlot.set_legend_box(ncol=ncol)

        self.__stepSizeControlPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        self.__stepSizeControlPlot.savefig(filenameSpinup)
        self.__stepSizeControlPlot.close_fig()


    def plotNormData(self, model, parameterId, norm='2', trajectory='', year=None, ncol=3, **kwargs):
        """
        Plot the relative error

        Plot the relative error of the spin up for the given biogeochemical
        model and parameterId. The plot includes the relative error for 
        different time steps. The tracer concentrations calculated with the
        spin up using the time step 1dt are used as reference solution.

        Parameters
        ----------
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
            additionalSimulationIds : list [tuple [int]], optional
                List for additional spin up plots using the simulationId and
                timestep defined in the tuples
            onlyAdditionalSimulationIds : bool, optional
                If True, plot only the norm of the simulations defined by
                the simulationIds in the additionalSimulationIds list
            filenamePrefix : str or None, optional
                Prefix of the filename for the figure

        NOTES
        -----
        The figure is saved in the directory defined in
        stepSizeControl.constants.PATH_FIGURE.
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, StepSizeControl_Constants.PARAMETERID_MAX+1)
        assert trajectory in ['', 'trajectory']
        assert norm in DB_Constants.NORM
        assert year is None or type(year) is int and 0 <= year
        assert type(ncol) is int and 0 < ncol

        #Parse keyword arguments
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.145, 'bottom': 0.165, 'right': 0.9625, 'top': 0.995}
        onlyAdditionalSimulationIds = kwargs['onlyAdditionalSimulationIds'] if 'onlyAdditionalSimulationIds' in kwargs else False
        additionalSimulationIds = kwargs['additionalSimulationIds'] if 'additionalSimulationIds' in kwargs and type(kwargs['additionalSimulationIds'] is list) else []
        filenameNorm = os.path.join(StepSizeControl_Constants.PATH_FIGURE, 'Norm', kwargs['filenamePrefix'] + StepSizeControl_Constants.PATTERN_FIGURE_NORM.format(trajectory, norm, model, parameterId) if 'filenamePrefix' in kwargs and type(kwargs['filenamePrefix']) is str else StepSizeControl_Constants.PATTERN_FIGURE_NORM.format(trajectory, norm, model, parameterId))
        kwargsLegend = {'handlelength': 1.0, 'handletextpad': 0.4, 'columnspacing': 1.0, 'borderaxespad': 0.3}

        #Create scatter plot
        self.__stepSizeControlPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        if onlyAdditionalSimulationIds:
            self.__stepSizeControlPlot.plot_tracer_norm_data(additionalSimulationIds[0][0], additionalSimulationIds[1:], ncol=ncol, year=year, **kwargsLegend)
        else:
            #TODO Setzen der notwednigen simulationIds und labels (abhaengig vom Modell und ParameterId)
            referenceSimulationId = 0
            simulationIds = [(608, '0.001'), (610, '0.0001')]
            self.__stepSizeControlPlot.plot_tracer_norm_data(referenceSimulationId, simulationIds, ncol=ncol, year=year, **kwargsLegend)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__stepSizeControlPlot.set_legend_box(ncol=ncol)

        self.__stepSizeControlPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])
        self.__stepSizeControlPlot.savefig(filenameNorm)
        self.__stepSizeControlPlot.close_fig()


    def plotScatterSpinupNorm(self, model, year=10000, norm='2', trajectory='', **kwargs):
        """
        Plot the spin up norm against the norm

        Plot the spin up norm value against the norm value for the given model
        year using the different yearIntervals and tolerances

        Parameters
        ----------
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

        NOTES
        -----
        The figure is saved in the directory defined in
        stepSizeControl.constants.PATH_FIGURE.
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

        filename = os.path.join(StepSizeControl_Constants.PATH_FIGURE, 'Norm', StepSizeControl_Constants.PATTERN_FIGURE_SPINUP_NORM.format(trajectory, norm, model))
        self.__stepSizeControlPlot.savefig(filename)
        self.__stepSizeControlPlot.close_fig()


    def plotScatterNormReduction(self, model, year=10000, norm='2', trajectory='', **kwargs):
        """
        Plot the relative error (norm) against the reduction

        Plot the norm norm value against the reduction of the computational
        costs for the given model year using the different yearIntervals and
        tolerances

        Parameters
        ----------
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

        filename = os.path.join(StepSizeControl_Constants.PATH_FIGURE, 'Norm', StepSizeControl_Constants.PATTERN_FIGURE_NORM_REDUCTION.format(trajectory, norm, model))
        self.__stepSizeControlPlot.savefig(filename)
        self.__stepSizeControlPlot.close_fig()


    def tableStepYear(self, stepYears=[1, 2, 5, 10, 25, 50], parameterId=0, year=10000, norm='2', tolerance=1.0):
        """
        Create table string for the results using different stepYears

        Create a string for a latex table including the relative errors (norm
        of the difference between the spin-up caluation using the step size
        control with step year and the reference solution calculated without
        step size control and time step 1dt) of the whole model hierarchy for
        every step year in the given list.

        Parameters
        ----------
        stepYears : list[int], default: [1, 2, 5, 10, 25, 50]
            List of the stepYears included in the table (a single row for each
            stepYear in the list)
        parameterId : int, default: 0
            Model parameterId
        year : int, default: 10000
            Model year of the tracer concentrations
        norm : str, default: '2'
            Used norm to compute the relative error
        tolerance : float, default: 1.0
            tolerance used for the step size control

        Returns
        -------
        str
            String of the latex table
        """
        assert type(stepYears) is list
        assert parameterId in range(0, StepSizeControl_Constants.PARAMETERID_MAX+1)
        assert type(year) is int and 0 < year
        assert norm in DB_Constants.NORM
        assert type(tolerance) is float and 0.0 < tolerance

        concentrationIdDic = {'N': 0, 'N-DOP': 1, 'NP-DOP': 2, 'NPZ-DOP': 3, 'NPZD-DOP': 4, 'MITgcm-PO4-DOP': 1}

        tableStr = '\\hline\nStep year & {:s}  \\\\\n\\hline\n'.format(' & '.join(map(str, Metos3d_Constants.METOS3D_MODELS)))
        for stepYear in stepYears:
            tableStr = tableStr + '{:>2d}'.format(stepYear)
            for model in Metos3d_Constants.METOS3D_MODELS:
                concentrationId = concentrationIdDic[model]
                #concentrationId = self.__stepSizeControlPlot._database.get_concentrationId_constantValues(model, Metos3d_Constants.INITIAL_CONCENTRATION[model])
                normValue = self.__stepSizeControlPlot._database.get_table_stepYear_value(model, parameterId, concentrationId, stepYear=stepYear, tolerance=tolerance, cpus=128, year=year, normError=norm, lhs=False)
                if len(normValue) == 1:
                    tableStr = tableStr + ' & {:.3e}'.format(normValue[0])
                else:
                    tableStr = tableStr + ' & {:^9s}'.format('-')
            tableStr = tableStr + ' \\\\\n'
        tableStr = tableStr + '\\hline\n'

        return tableStr



if __name__ == '__main__':
    main()

