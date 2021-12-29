#!/usr/bin/env python
# -*- coding: utf8 -*

import matplotlib.pyplot as plt
import numpy as np
import os

import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.petsc.petscfile as petsc
from metos3dutil.plot.plot import Plot
import decreasingTimesteps.constants as DecreasingTimesteps_Constants
from decreasingTimesteps.DecreasingTimestepsDatabase import DecreasingTimestepsDatabase


class DecreasingTimestepsPlot(Plot):
    """
    Creation of plots using decreasing time steps

    Creation of plots for the spin up calculation using decreasing time steps

    Attributes
    ----------
    colorsTimestep : dict [int, str]
        Assignment of timesteps to different colors used in the plot
    """

    def __init__(self, orientation='gmd', fontsize=8, dbpath=DecreasingTimesteps_Constants.DB_PATH, cmap=None, completeTable=True):
        """
        Constructs the environment to plot the data using decreasing time steps

        Parameters
        ----------
        orientation : str, default: gmd
            Orientation of the figure
        fontsize : int, default: 8
            Fontsize used in the figure
        dbpath : str, default: decreasingTimesteps.constants.DB_PATH
            Path to the sqlite database
        cmap : matplotlib.colors.Colormap or None, default: None
            Colormap used in the surface plot to visualize tracer
            concentrations
        completeTable : bool, default: True
            If the value is True, use all columns (even columns with value
            None) in SELECT queries on the database
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        Plot.__init__(self, cmap=cmap, orientation=orientation, fontsize=fontsize)
        self._database = DecreasingTimestepsDatabase(dbpath=dbpath, completeTable=completeTable)

        self._colorsTimestep = {1: 'C0', 2: 'C1', 4: 'C2', 8: 'C3', 16: 'C4', 32: 'C5', 64: 'C9'}


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self._database.close_connection()


    def plot_spinup_data(self, ncol=3, simulationIds=[], subPlot=False, axesResultSmall=[.61, .30, .3, .34], subPlotModelYear=8000, **kwargs):
        """
        Plot the spinup for the given simulationIds

        Parameters
        ----------
        ncol : int, default: 3
            Number of columns for the legend
        simulationIds : list [tuple], default: []
            List for additional spin up plots using the simulationId and
            label defined in the tuples
        subPlot : bool, default: False
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added
        axesResultSmall : list [float], default: [.61, .30, .3, .34]
            Dimensions of the subplot
        subPlotModelYear : int, default: 8000
            Start model year for the subplot
        **kwargs : dict
            Additional keyword arguments with keys:

            handlelength : float
                The length of the legend handles, in font-size units.
            handletextpad : float
                The pad between the legend handle and text, in font-size units.
            columnspacing : float
                The spacing between columns, in font-size units.
            borderaxespad : float
                The pad between the axes and legend border, in font-size units.
        """
        assert type(ncol) is int and 0 < ncol
        assert type(simulationIds) is list
        assert type(subPlot) is bool
        assert type(axesResultSmall) is list and len(axesResultSmall) == 4
        assert type(subPlotModelYear) is int and 0 <= subPlotModelYear

        colorIndexList = [-1, 0, 2, 1, 3, 4, 5, 6, 7, 8]
        colorIndex = 0

        if subPlot:
            self.__axesResultSmall = plt.axes(axesResultSmall)

        #Plot the spin up for the simulationIds
        for simulationId, label in simulationIds:
            if self._database.get_convergence(simulationId):
                data = self._database.read_spinup_values_for_simid(simulationId)
                try:
                    self._axesResult.plot(data[:,0], data[:,1], label=label, color=self._colors[colorIndexList[colorIndex]])
                    if subPlot:
                        self.__axesResultSmall.plot(data[subPlotModelYear:,0], data[subPlotModelYear:,1], color=self._colors[colorIndexList[colorIndex]])
                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure with was not created.")
                colorIndex += 1

        #Set labels
        self._axesResult.set_xlabel(r'Model years [\si{{\Modelyear}}]')
        self._axesResult.set_ylabel(r'Norm [\si{\milli\mole\Phosphat\per\cubic\meter}]')
        self._axesResult.set_yscale('log', basey=10)
        self._axesResult.legend(loc='best', ncol=ncol, handlelength=kwargs['handlelength'], handletextpad=kwargs['handletextpad'], columnspacing=kwargs['columnspacing'], borderaxespad=kwargs['borderaxespad'])


    def plot_tracer_norm_data(self, referenceSimulationId, simulationIds, norm='2', trajectory='', year=None, ncol=3, **kwargs):
        """
        Plot the norm of the tracer concentration difference
        
        Plot the development over the spin up (10000 years) of the difference
        between the 1dt solution and solutions calculated with decreasing time
        steps in the norm for the given simulationIds.

        Parameters
        ----------
        referenceSimulationId : int
            SimulationId of the the reference solution.
        simulationIds : list [tuple]
            List of simulations using the simulationId and label defined in the
            tuples.
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        year : int, default: None
            Use the reference solution (1dt solution) at the given year (e.g.
            the reference solution after a spin up over 10000 model years). If
            the value is None, use the same year for the reference and other
            solution.
        ncol : int, default: 3
            Number of columns for the legend
        **kwargs : dict
            Additional keyword arguments with keys:

            handlelength : float
                The length of the legend handles, in font-size units.
            handletextpad : float
                The pad between the legend handle and text, in font-size units.
            columnspacing : float
                The spacing between columns, in font-size units.
            borderaxespad : float
                The pad between the axes and legend border, in font-size units.
        """
        assert type(referenceSimulationId) is int and 0 <= referenceSimulationId
        assert type(simulationIds) is list
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'trajectory']
        assert year is None or type(year) is int and 0 <= year
        assert type(ncol) is int and 0 < ncol

        colorIndexList = [0, 2, 1, 3, 4, 5, 6, 7, 8]
        colorIndex = 0

        #Get reference solution
        if year is None:
            data_solution = self._database.read_tracer_norm_values_for_simid(referenceSimulationId, norm=norm, trajectory=trajectory)
        else:
            data_solution = self._database.read_tracer_norm_value_for_simid_year(referenceSimulationId, year, norm=norm, trajectory=trajectory)

        #Plot the norm for the extra simulationIds
        for simulationId, label in simulationIds:
            if self._database.get_convergence(simulationId):
                data = self._database.read_tracer_difference_norm_values_for_simid(simulationId, referenceSimulationId, yearB=year, norm=norm, trajectory=trajectory)
                if year is None:
                    print('SimulationId: {}\nNorm: {}\ntracerNormShapes: {} {}'.format(simulationId, norm, np.shape(data), np.shape(data_solution)))
                    data[:,1] = data[:,1] / data_solution[:,1]
                else:
                    data[:,1] = data[:,1] / data_solution
                try:
                    self._axesResult.plot(data[:,0], data[:,1], color=self._colors[colorIndexList[colorIndex]], label=label)
                    colorIndex += 1

                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure with was not created.")

        #Set labels
        self._axesResult.set_xlabel(r'Model years [\si{{\Modelyear}}]')
        self._axesResult.set_ylabel(r'Relative error')
        self._axesResult.set_yscale('log', basey=10)
        if 'legend' in kwargs and kwargs['legend']:
            self._axesResult.legend(loc='best', ncol=ncol, handlelength=kwargs['handlelength'], handletextpad=kwargs['handletextpad'], columnspacing=kwargs['columnspacing'], borderaxespad=kwargs['borderaxespad'])

    
    def plot_scatter_spinup_norm(self, model, timestep=64, year=10000, norm='2', trajectory='', yearIntervalList=[50], toleranceList=[0.001], alpha=0.75, locMarkerBox='lower right', toleranceLegend=False, **kwargs):
        """
        Scatter plot of the spin up against the norm

        Plot a scatter plot of the relation between the spin up norm and the
        norm of the tracer difference between the spin up calculation using
        decreasing time steps and the spin up calculation using the time step
        1dt. The plot visualizes the ratio for the given yearInterval using
        different colors and for the given tolerance using different markers.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 64
            Initial time step of the decreasing time step simulation
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'Trajectory' the norm over the whole trajectory
        yearIntervalList : list [int], default: [50]
            Representation of the relation between spin up norm and the norm of
            tracer difference for each yearInterval used in the decreasing
            time steps simulation
        toleranceList : list [float], default: [0.001]
            Representation of the relation between spin up norm and the norm of
            tracer difference for each tolerance
        alpha : float, default: 0.75
            The alpha blending value of the scatter plot, between 0
            (transparent) and 1 (opaque)
        locMarkerBox : {'best', 'upper right', 'upper left', 'lower left',
                        'lower right', 'right', 'center left', 'center right',
                        'lower center', 'upper center', 'center'},
                      default: 'lower right'
            Location of the legend including the different marker
        toleranceLegend : bool, default: True
            If True, create the legend for the different tolerances
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(yearIntervalList) is list
        assert type(toleranceList) is list
        assert type(alpha) is float and 0.0 <= alpha and alpha <= 1.0
        assert locMarkerBox in ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']
        assert type(toleranceLegend) is bool

        markerList = ['.', '1', 'x', '3', '+', '2', '4']

        colorIndex = 0
        for yearInterval in yearIntervalList:
            markerIndex = 0
            plotList = []
            for tolerance in toleranceList:
                data = self._database.read_spinupNorm_relNorm_for_model_year(model, timestep=timestep, yearInterval=yearInterval, tolerance=tolerance, year=year, norm=norm, trajectory=trajectory, lhs=False)

                try:
                    if markerIndex == 0:
                        p1 = self._axesResult.scatter(data[:,2], data[:,3], s=4, marker=markerList[markerIndex], color=self._colors[colorIndex], alpha=alpha, label = r'{}'.format(yearInterval))
                    else:
                        plotList.append(self._axesResult.scatter(data[:,2], data[:,3], s=4, marker=markerList[markerIndex], color=self._colors[colorIndex], alpha=alpha))

                    #Create legend for different markers
                    if len(toleranceList) > 1 and toleranceLegend and len(toleranceList) == markerIndex+1:
                        legendTolerance = self._axesResult.legend([p1] + plotList, toleranceList[:len(plotList) + 1], loc=locMarkerBox, title='Tolerance', borderaxespad=0.2, labelspacing=0.2, borderpad=0.2, handlelength=0.4, handletextpad=0.5)

                        #Change color of the markers to black
                        lh = legendTolerance.legendHandles
                        for i in range(len(lh)):
                            lh[i].set_color('black')

                        self._axesResult.add_artist(legendTolerance)
                        toleranceLegend = False

                    self._axesResult.set_xscale('log', basex=10)
                    self._axesResult.set_yscale('log', basey=10)
                    self._axesResult.set_xlabel(r'Norm [\si{\milli\mole\Phosphat\per\cubic\meter}]')
                    self._axesResult.set_ylabel(r'Relative error')
                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure was not created.")

                markerIndex += 1
            colorIndex += 1

        if 'xticksminor' in kwargs and 'xticksminorlabel' in kwargs:
            self._axesResult.set_xticks([3*10**(-5), 4*10**(-5), 5*10**(-5), 6*10**(-5), 7*10**(-5), 8*10**(-5), 9*10**(-5)], minor=True)
            self._axesResult.set_xticklabels([r'$3 \times 10^{-5}$', '', '', r'$6 \times 10^{-5}$', '', '', ''], minor=True)


    def plot_scatter_norm_costreduction(self, model, timestep=64, year=10000, norm='2', trajectory='', yearIntervalList=[50], toleranceList=[0.001], alpha=0.75, locMarkerBox='lower right', toleranceLegend=True, **kwargs):
        """
        Scatter plot of the norm against the cost savings

        Plot a scatter plot of the relation between the norm of the tracer
        difference between the spin up calculation using decreasing time steps
        and the spin up calculation using the time step 1dt as well as the cost
        saving. The plot visualizes the ratio for the given yearInterval using
        different colors and for the given tolerance using different markers.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 64
            Initial time step of the decreasing time step simulation
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'Trajectory' the norm over the whole trajectory
        yearIntervalList : list [int], default: [50]
            Representation of the relation between spin up norm and the norm of
            tracer difference for each yearInterval used in the decreasing
            time steps simulation
        toleranceList : list [float], default: [0.001]
            Representation of the relation between spin up norm and the norm of
            tracer difference for each tolerance
        alpha : float, default: 0.75
            The alpha blending value of the scatter plot, between 0
            (transparent) and 1 (opaque)
        locMarkerBox : {'best', 'upper right', 'upper left', 'lower left',
                        'lower right', 'right', 'center left', 'center right',
                        'lower center', 'upper center', 'center'},
                      default: 'lower right'
            Location of the legend including the different marker
        toleranceLegend : bool, default: True
            If True, create the legend for the different tolerances
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(yearIntervalList) is list
        assert type(toleranceList) is list
        assert type(alpha) is float and 0.0 <= alpha and alpha <= 1.0
        assert locMarkerBox in ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']
        assert type(toleranceLegend) is bool

        markerList = ['.', '1', 'x', '3', '+', '2', '4']

        colorIndex = 0
        for yearInterval in yearIntervalList:
            markerIndex = 0
            plotList = []
            for tolerance in toleranceList:
                data = self._database.read_spinupNorm_relNorm_for_model_year(model, timestep=timestep, yearInterval=yearInterval, tolerance=tolerance, year=year, norm=norm, trajectory=trajectory, lhs=False)

                reductionData = np.zeros(shape=(len(data), 2))
                #Set relative error
                reductionData[:,0] = data[:,3]

                #Calculate reduction of the computational costs
                i = 0
                for simulationId in data[:,0]:
                    reductionData[i,1] = self._calculate_reduction(int(simulationId), yearInterval)
                    i += 1

                try:
                    if markerIndex == 0:
                        p1 = self._axesResult.scatter(reductionData[:,0], reductionData[:,1], s=4, marker=markerList[markerIndex], color=self._colors[colorIndex], alpha=alpha, label = r'{}'.format(yearInterval))
                    else:
                        plotList.append(self._axesResult.scatter(reductionData[:,0], reductionData[:,1], s=4, marker=markerList[markerIndex], color=self._colors[colorIndex], alpha=alpha))

                    #Create legend for different markers
                    if len(toleranceList) > 1 and toleranceLegend and len(toleranceList) == markerIndex+1:
                        legendTolerance = self._axesResult.legend([p1] + plotList, toleranceList[:len(plotList) + 1], loc=locMarkerBox, title='Tolerance', borderaxespad=0.2, labelspacing=0.2, borderpad=0.2, handlelength=0.4, handletextpad=0.5)

                        #Change color of the markers to black
                        lh = legendTolerance.legendHandles
                        for i in range(len(lh)):
                            lh[i].set_color('black')

                        self._axesResult.add_artist(legendTolerance)
                        toleranceLegend = False

                    self._axesResult.set_xscale('log', basex=10)
                    self._axesResult.set_xlabel(r'Relative error')
                    self._axesResult.set_ylabel(r'Cost saving [\%]')
                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure was not created.")

                markerIndex += 1
            colorIndex += 1


    def _calculate_reduction(self, simulationId, yearInterval):
        """
        Returns the reduction of the computational costs in percent

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        yearInterval : int
            YearInterval used in the decreasing time steps simulation

        Returns
        -------
        float
            Reduction of the computational costs
        """
        assert type(simulationId) is int and simulationId >= 0

        timestepData = self._database.read_year_timesteps(simulationId)
        evaluationsReference = Metos3d_Constants.METOS3D_STEPS_PER_YEAR * 10000

        evaluations = 0
        for i in range(len(timestepData)):
            evaluations += yearInterval * Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestepData[i,1]

        return float(100.0 * (evaluationsReference - evaluations) / evaluationsReference)

