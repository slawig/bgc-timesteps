#!/usr/bin/env python
# -*- coding: utf8 -*

import matplotlib.pyplot as plt
import numpy as np
import os

import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.plot.plot import Plot
import stepSizeControl.constants as StepSizeControl_Constants
from stepSizeControl.StepSizeControlDatabase import StepSizeControlDatabase


class StepSizeControlPlot(Plot):
    """
    Creation of plots using the step size control

    Attributes
    ----------
    colorsTimestep : dict [int, str]
        Assignment of timesteps to different colors used in the plot
    """

    def __init__(self, orientation='gmd', fontsize=8, dbpath=StepSizeControl_Constants.DB_PATH, cmap=None, completeTable=True):
        """
        Constructs the environment to plot the data using the step size control

        Parameter
        ----------
        orientation : str, default: gmd
            Orientation of the figure
        fontsize : int, default: 8
            Fontsize used in the figure
        dbpath : str, default: stepSizeControl.constants.DB_PATH
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
        self._database = StepSizeControlDatabase(dbpath=dbpath, completeTable=completeTable)

        self._colorsTimestep = {0: 'k', 1: 'C0', 2: 'C1', 4: 'C2', 8: 'C3', 16: 'C4', 32: 'C5', 64: 'C9'}


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self._database.close_connection()


    def plot_spinup_data(self, ncol=3, simulationIds=[], subPlot=False, axesResultSmall=[.61, .30, .3, .34], subPlotModelYear=8000, **kwargs):
        """
        Plot the spinup for the given simulationIds

        Parameter
        ---------
        ncol : int, default: 3
            Number of columns for the legend
        simulationIds : list [tuple], default: []
            List for additional spin up plots using the simulationId, a
            reference flag and label defined in the tuples
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

        if subPlot:
            self.__axesResultSmall = plt.axes(axesResultSmall)

        colorIndex = -1
        #Plot the spin up for the simulationIds
        for simulationId, reference, label in simulationIds:
            data = self._database.read_spinup_values_for_simid(simulationId, reference=reference)
            try:
                self._axesResult.plot(data[:,0], data[:,1], color=self._colors[colorIndex], label=label)
                if subPlot:
                    self.__axesResultSmall.plot(data[subPlotModelYear:,0], data[subPlotModelYear:,1])
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Figure with was not created.")
            colorIndex += 1

        #Set labels
        self._axesResult.set_xlabel(r'Model years [\si{{\Modelyear}}]')
        self._axesResult.set_ylabel(r'Norm [\si{\milli\mole\Phosphat\per\cubic\meter}]')
        self._axesResult.set_yscale('log', basey=10)
        self._axesResult.legend(loc='best', ncol=ncol, labelspacing=kwargs['labelspacing'], handlelength=kwargs['handlelength'], handletextpad=kwargs['handletextpad'], columnspacing=kwargs['columnspacing'], borderaxespad=kwargs['borderaxespad'], borderpad=kwargs['borderpad'])


    def plot_tracer_norm_data(self, referenceSimulationId, simulationIds, norm='2', trajectory='', year=None, ncol=3, **kwargs):
        """
        Plot the norm of the tracer concentration difference
        
        Plot the development over the spin up (10000 years) of the difference
        between the 1dt solution and solutions calculated with the step size
        control in the norm for the given simulationIds.

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

        colorIndexList = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        colorIndex = 0

        #Get reference solution
        if year is None:
            data_solution = self._database.read_tracer_norm_values_for_simid(referenceSimulationId, norm=norm, trajectory=trajectory)
        else:
            data_solution = self._database.read_tracer_norm_value_for_simid_year(referenceSimulationId, year, norm=norm, trajectory=trajectory)

        #Plot the norm for the extra simulationIds
        for (simulationId, label) in simulationIds:
            data = self._database.read_tracer_difference_norm_values_for_simid(simulationId, referenceSimulationId, yearB=year, norm=norm, trajectory=trajectory)
            if year is None:
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


    def plot_scatter_spinup_norm(self, model, stepSizeControlParameter, year=10000, norm='2', trajectory='', alpha=0.75, **kwargs):
        """
        Scatter plot of the spin up against the norm

        Plot a scatter plot of the relation between the spin up norm and the
        norm of the tracer difference between the spin up calculation using
        step size control and the spin up calculation using the time step
        1dt. The plot visualizes the ratio for the given step size control
        configuration using different colors.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        stepSizeControlParameter : list[tuple]
            List of tuple with the configurations of the step size control
            consisting of (startTimestep, stepYear, tolerance, rho, eta, cpus,
            norm, checkConcentration, singleStep, singleStepYear, label)
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'Trajectory' the norm over the whole trajectory
        alpha : float, default: 0.75
            The alpha blending value of the scatter plot, between 0
            (transparent) and 1 (opaque)
        **kwargs : dict
            Additional keyword arguments with keys:

            'xticksminor' : list [float]
                List of positions for the x minor ticks
            'xticksminorlabel' : list [str]
                List of labels for the x minor ticks
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(stepSizeControlParameter) is list
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(alpha) is float and 0.0 <= alpha and alpha <= 1.0

        colorIndex = 0
        for (startTimestep, stepYear, tolerance, rho, eta, cpus, normStepSizeControl, checkConcentration, singleStep, singleStepYear, label) in stepSizeControlParameter:
            data = self._database.read_spinupNorm_relNorm_for_model_year(model, startTimestep=startTimestep, stepYear=stepYear, tolerance=tolerance, rho=rho, eta=eta, cpus=cpus, normStepSizeControl=normStepSizeControl, checkConcentration=checkConcentration, singleStep=singleStep, singleStepYear=singleStepYear, year=year, norm=norm, trajectory=trajectory, lhs=False)

            try:
                self._axesResult.scatter(data[:,2], data[:,3], s=4, color=self._colors[colorIndex], alpha=alpha, label = r'{}'.format(label))
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Figure was not created.")

            colorIndex += 1

        self._axesResult.set_xscale('log', basex=10)
        self._axesResult.set_yscale('log', basey=10)
        self._axesResult.set_xlabel(r'Norm [\si{\milli\mole\Phosphat\per\cubic\meter}]')
        self._axesResult.set_ylabel(r'Relative error')

        if 'xticksminor' in kwargs and 'xticksminorlabel' in kwargs:
            self._axesResult.set_xticks(kwargs['xticksminor'], minor=True)
            self._axesResult.set_xticklabels(kwargs['xticksminorlabel'], minor=True)


    def plot_scatter_norm_costreduction(self, model, stepSizeControlParameter, year=10000, norm='2', trajectory='', alpha=0.75, **kwargs):
        """
        Scatter plot of the norm against the cost saving

        Plot a scatter plot of the relation between the cost saving and the
        norm of the tracer difference between the spin up calculation using
        step size control and the spin up calculation using the time step
        1dt. The plot visualizes the ratio for the given step size control
        configuration using different colors.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        stepSizeControlParameter : list[tuple]
            List of tuple with the configurations of the step size control
            consisting of (startTimestep, stepYear, tolerance, rho, eta, cpus,
            norm, checkConcentration, singleStep, singleStepYear, label)
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'Trajectory' the norm over the whole trajectory
        alpha : float, default: 0.75
            The alpha blending value of the scatter plot, between 0
            (transparent) and 1 (opaque)
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(stepSizeControlParameter) is list
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(alpha) is float and 0.0 <= alpha and alpha <= 1.0

        colorIndex = 0
        for (startTimestep, stepYear, tolerance, rho, eta, cpus, normStepSizeControl, checkConcentration, singleStep, singleStepYear, label) in stepSizeControlParameter:
            data = self._database.read_spinupNorm_relNorm_for_model_year(model, startTimestep=startTimestep, stepYear=stepYear, tolerance=tolerance, rho=rho, eta=eta, cpus=cpus, normStepSizeControl=normStepSizeControl, checkConcentration=checkConcentration, singleStep=singleStep, singleStepYear=singleStepYear, year=year, norm=norm, trajectory=trajectory, lhs=False)

            reductionData = np.zeros(shape=(len(data), 2))
            #Set relative error
            reductionData[:,0] = data[:,3]

            #Calculate reduction of the computational costs
            i = 0
            for simulationId in data[:,0]:
                reductionData[i,1] = self._calculate_reduction(int(simulationId), stepYear)
                i += 1

            try:
                self._axesResult.scatter(reductionData[:,0], reductionData[:,1], s=4, color=self._colors[colorIndex], alpha=alpha, label = r'{}'.format(label))
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Figure was not created.")

            colorIndex += 1

        self._axesResult.set_xscale('log', basex=10)
        self._axesResult.set_xlabel(r'Relative error')
        self._axesResult.set_ylabel(r'Cost saving [\%]')


    def _calculate_reduction(self, simulationId, stepYear):
        """
        Returns the reduction of the computational costs in percent

        Parameter
        ---------
        simulationId : int
            Id defining the parameter for spin up calculation
        stepYear : int
            Number of model years without adapting the time step

        Returns
        -------
        float
            Reduction of the computational costs
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(stepYear) is int and stepYear > 0

        timestepData = self._database.read_stepControl_values_for_simulationId(simulationId)
        evaluationsReference = Metos3d_Constants.METOS3D_STEPS_PER_YEAR * 10000

        evaluations = 0
        for i in range(len(timestepData)):
            #For each model year, evaluations using the small time step (if the step was accepted) plus the evaluations using the big time step
            if bool(timestepData[i,3]):
                evaluations += stepYear * (Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestepData[i,1])
            evaluations += stepYear * (Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestepData[i,2])

        return float(100.0 * (evaluationsReference - evaluations) / evaluationsReference)

