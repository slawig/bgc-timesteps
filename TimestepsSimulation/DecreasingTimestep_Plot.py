#!/usr/bin/env python
# -*- coding: utf8 -*

import numpy as np
import os
import matplotlib.pyplot as plt

import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import decreasingTimesteps.constants as DecreasingTimesteps_Constants
from decreasingTimesteps.DecreasingTimestepsPlot import DecreasingTimestepsPlot


def main(orientation='gmd', fontsize=8, plotSpinup=False, plotNorm=False, plotRelationSpinupNorm=True, plotRelationNormReduction=True):
    """
    Plot the results using decreasing time steps for the spin up

    Create plots of the results using decreasing time steps for the spin up
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

    decreasingTimestepsPlot = DecreasingTimestepsPlots(orientation=orientation, fontsize=fontsize)

    #Spin up norm plot
    if plotSpinup:
        parameterId = 0
        kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
        kwargs['N'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(0, 'Reference'), (608, '0.001'), (610, '0.0001')], 'subPlotModelYear': 9000, 'axesResultSmall': [.69, .4, .2, .2]}
        kwargs['N-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(1, '1dt'), (615, '0.001'), (617, '0.0001')], 'subPlotModelYear': 9000}
        kwargs['NP-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(2, '1dt'), (622, '0.001'), (624, '0.0001')], 'subPlotModelYear': 9000}
        kwargs['NPZ-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(3, '1dt'), (629, '0.001'), (631, '0.0001')], 'subPlotModelYear': 9000}
        kwargs['NPZD-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(4, '1dt'), (636, '0.001'), (638, '0.0001')], 'subPlotModelYear': 9000}
        kwargs['MITgcm-PO4-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(5, '1dt'), (643, '0.001'), (645, '0.0001')], 'subPlotModelYear': 9000}

        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            decreasingTimestepsPlot.plotSpinupData(metos3dModel, parameterId, **kwargs[metos3dModel])


    #Plot of the norm data
    if plotNorm:
        year = 10000
        kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
        kwargs['N'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(0, '1dt'), (608, '0.001'), (610, '0.0001')], 'subPlotModelYear': 9000}
        #kwargs['N'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(6, '1dt'), (672, '0.001'), (673, '0.0001')], 'subPlotModelYear': 9000}
        kwargs['N-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(1, '1dt'), (615, '0.001'), (617, '0.0001')], 'subPlotModelYear': 9000}
        kwargs['NP-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(1, '1dt'), (622, '0.001'), (624, '0.0001')], 'subPlotModelYear': 9000}
        kwargs['NPZ-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(1, '1dt'), (629, '0.001'), (631, '0.0001')], 'subPlotModelYear': 9000}
        kwargs['NPZD-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(1, '1dt'), (636, '0.001'), (638, '0.0001')], 'subPlotModelYear': 9000}
        kwargs['MITgcm-PO4-DOP'] = {'onlyAdditionalSimulationIds': True, 'additionalSimulationIds': [(1, '1dt'), (643, '0.001'), (645, '0.0001')], 'subPlotModelYear': 9000}

        for norm in ['2', 'BoxweightedVol']:
            for parameterId in range(0, 1): #Timesteps_Constants.PARAMETERID_MAX+1):
                for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
                    decreasingTimestepsPlot.plotNormData(metos3dModel, parameterId, norm=norm,  year=year, **kwargs[metos3dModel])


    #Plot relation between spin up norm and relative error
    if plotRelationSpinupNorm:
        year = 10000
        kwargs = {}
        kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
        kwargs['N'] = {'locMarkerBox' : 'upper left', 'toleranceLegend': True, 'legend_box': True, 'xticksminor': [3*10**(-5), 4*10**(-5), 5*10**(-5), 6*10**(-5), 7*10**(-5), 8*10**(-5), 9*10**(-5)], 'xticksminorlabel': [r'$3 \times 10^{-5}$', '', '', r'$6 \times 10^{-5}$', '', '', '']}
        kwargs['N-DOP'] = {'subplot_adjust': {'left': 0.145, 'bottom': 0.17, 'right': 0.9825, 'top': 0.9025}}
        for metos3dModel in  Metos3d_Constants.METOS3D_MODELS:
            decreasingTimestepsPlot.plotScatterSpinupNorm(metos3dModel, year=year, **kwargs[metos3dModel])


    #Plot relation between relative error and the reduction of the computational costs
    if plotRelationNormReduction:
        year = 10000
        kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'NPZ-DOP': {}, 'NPZD-DOP': {}, 'MITgcm-PO4-DOP': {}}
        kwargs['N'] = {'locMarkerBox' : 'lower right', 'legend_box': True, 'toleranceLegend': True}
        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            decreasingTimestepsPlot.plotScatterNormReduction(metos3dModel, year=year, **kwargs[metos3dModel])

    decreasingTimestepsPlot.closeDatabaseConnection()



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

        self.__decreasingTimestepPlot = DecreasingTimestepsPlot(orientation=self._orientation, fontsize=self._fontsize, dbpath='/gxfs_work1/cau/sunip350/metos3d/LatinHypercubeSample/Database/20210909_DecreasingTimesteps_Database.db')


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self.__decreasingTimestepPlot.closeDatabaseConnection()



    def plotSpinupData(self, model, parameterId, ncol=3, subPlot=True, **kwargs):
        """
        Plot the spin up norm.

        Plot the spin up norm for the given biogeochemical model and
        parameterId. The plot includes the spin up norm for the reference
        simulation (using 1dt) and the simulation using the decreasing time
        steps.

        Parameters
        ----------
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
        decreasingTimesteps.constants.PATH_FIGURE.
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
        onlyAdditionalSimulationIds = kwargs['onlyAdditionalSimulationIds'] if 'onlyAdditionalSimulationIds' in kwargs else False
        additionalSimulationIds = kwargs['additionalSimulationIds'] if 'additionalSimulationIds' in kwargs and type(kwargs['additionalSimulationIds'] is list)else []
        filenameSpinup = os.path.join(DecreasingTimesteps_Constants.PATH_FIGURE, 'Spinup', model, kwargs['filenamePrefix'] + DecreasingTimesteps_Constants.PATTERN_FIGURE_SPINUP.format(model, parameterId) if 'filenamePrefix' in kwargs and type(kwargs['filenamePrefix']) is str else DecreasingTimesteps_Constants.PATTERN_FIGURE_SPINUP.format(model, parameterId))
        kwargsLegend = {'handlelength': 1.0, 'handletextpad': 0.4, 'columnspacing': 1.0, 'borderaxespad': 0.3}

        #Create scatter plot
        self.__decreasingTimestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        if onlyAdditionalSimulationIds:
            self.__decreasingTimestepPlot.plot_spinup_data(ncol=ncol, simulationIds=additionalSimulationIds, subPlot=subPlot, axesResultSmall=axesResultSmall, subPlotModelYear=subPlotModelYear, **kwargsLegend)
        else:
            #TODO Setzen der notwednigen simulationIds und labels (abhaengig vom Modell und ParameterId)
            self.__decreasingTimestepPlot.plot_spinup_data(ncol=ncol, simulationIds=additionalSimulationIds)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__decreasingTimestepPlot.set_legend_box(ncol=ncol)

        self.__decreasingTimestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        self.__decreasingTimestepPlot.savefig(filenameSpinup)
        self.__decreasingTimestepPlot.close_fig()


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

            timestepList : list [int], optional
                Representation of the spin up norm for each specified timestep
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
        timesteps.constants.PATH_FIGURE.
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
        onlyAdditionalSimulationIds = kwargs['onlyAdditionalSimulationIds'] if 'onlyAdditionalSimulationIds' in kwargs else False
        additionalSimulationIds = kwargs['additionalSimulationIds'] if 'additionalSimulationIds' in kwargs and type(kwargs['additionalSimulationIds'] is list)else []
        filenameNorm = os.path.join(DecreasingTimesteps_Constants.PATH_FIGURE, 'Norm', kwargs['filenamePrefix'] + DecreasingTimesteps_Constants.PATTERN_FIGURE_NORM.format(trajectory, norm, model, parameterId) if 'filenamePrefix' in kwargs and type(kwargs['filenamePrefix']) is str else DecreasingTimesteps_Constants.PATTERN_FIGURE_NORM.format(trajectory, norm, model, parameterId))
        kwargsLegend = {'handlelength': 1.0, 'handletextpad': 0.4, 'columnspacing': 1.0, 'borderaxespad': 0.3}

        
        #Create scatter plot
        self.__decreasingTimestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        if onlyAdditionalSimulationIds:
            self.__decreasingTimestepPlot.plot_tracer_norm_data(additionalSimulationIds[0][0], additionalSimulationIds[1:], ncol=ncol, year=year, **kwargsLegend)
        else:
            #TODO Setzen der notwednigen simulationIds und labels (abhaengig vom Modell und ParameterId)
            referenceSimulationId = 0
            simulationIds = [(608, '0.001'), (610, '0.0001')]
            self.__decreasingTimestepPlot.plot_tracer_norm_data(referenceSimulationId, simulationIds, ncol=ncol, year=year, **kwargsLegend)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__decreasingTimestepPlot.set_legend_box(ncol=ncol)

        self.__decreasingTimestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        self.__decreasingTimestepPlot.savefig(filenameNorm)
        self.__decreasingTimestepPlot.close_fig()


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
        decreasingTimesteps.constants.PATH_FIGURE.
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

        filename = os.path.join(DecreasingTimesteps_Constants.PATH_FIGURE, 'Norm', DecreasingTimesteps_Constants.PATTERN_FIGURE_SPINUP_NORM.format(trajectory, norm, model))
        self.__decreasingTimestepPlot.savefig(filename)
        self.__decreasingTimestepPlot.close_fig()


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

        filename = os.path.join(DecreasingTimesteps_Constants.PATH_FIGURE, 'Norm', DecreasingTimesteps_Constants.PATTERN_FIGURE_NORM_REDUCTION.format(trajectory, norm, model))
        self.__decreasingTimestepPlot.savefig(filename)
        self.__decreasingTimestepPlot.close_fig()




if __name__ == '__main__':
    main()

