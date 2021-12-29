#!/usr/bin/env python
# -*- coding: utf8 -*

import gc
import logging
import multiprocessing as mp
import numpy as np
import os
import re
import shutil
import sqlite3
import time
import threading

import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.metos3d.Metos3d import Metos3d, readBoxVolumes
from metos3dutil.simulation.AbstractClassSimulation import AbstractClassSimulation
import neshCluster.constants as NeshCluster_Constants
import decreasingTimesteps.constants as DecreasingTimesteps_Constants
from decreasingTimesteps.DecreasingTimestepsDatabase import DecreasingTimestepsDatabase
import timesteps.constants as Timesteps_Constants


class DecreasingTimestepsSimulation(AbstractClassSimulation):
    """
    Class for the simulation using decreasing time steps
    """

    def __init__(self, metos3dModel, parameterId=0, timestep=64, tolerance=0.01, yearInterval=50, logfile=None):
        """
        Initializes the simulation using decreasing time steps

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int, default: 0
            Id of the parameter of the latin hypercube example
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 64
            Initial time step of the spin up simulation
        tolerance : float, default: 0.01
            Decrease the time step if the relative error of the last time
            interval is less than the tolerance
        yearInterval : int, default: 50
            Test the reduction of the spin up simulation (relative error) after
            the given number of years
        logfile : None or str, default: None
            Name and path of the logfile

        Attributes
        ----------
        _database
            Database connection inherited from the module 
            metos3dutil.database.DatabaseMetos3d
        _overwrite : bool, default: True
            Flag for the insert into the database. If True, overwrite existing
            database entries
        _metos3dModel : str
            Name of the biogeochemical model
        _parameterId : int
            Id of the parameter of the latin hypercube example
        _timestep : {1, 2, 4, 8, 16, 32, 64}
            Time step of the spin up simulation
        _concentrationId : int
            Id of the initial concentration
        _simulationId : int
            Id identifying the simulation in the datbase
        _modelParameter: list [float]
            List with the constant initial concentration
        _path : str
            Path of the simulation directory
        _years : int
            Model years of the spin up
        _trajectoryYear : int
            Interval saving the tracer concentration during the spin up
        _lastSpinupYear : int
            Number of model years of the spin up
        _spinupTolerance : float or None
            Tolerance of the spin up
        _trajectoryFlag : bool
            If True, calculate the trajectory for the evaluation
        _removeTracer : bool
            If True, remove the tracer after the one step calculation
        _nodes : int
            Number of nodes for the calculation on the high performance
            cluster
        _defaultConcentration : bool
            If True, uses standard constant initial concentration
        _yearInterval : int
            Number of years after which the relative error is checked to
            decrease the time step
        _tolerance : float
            Decrease the time step if the relative error of the last time
            interval is less than the tolerance
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(DecreasingTimesteps_Constants.PARAMETERID_MAX+1)
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(tolerance) is float and 0 < tolerance
        assert type(yearInterval) is int and 0 < yearInterval
        assert logfile is None or os.path.exists(logfile) and os.path.isfile(logfile)

        #Time
        startTime = time.time()

        #Model parameter
        self._yearInterval = yearInterval
        self._tolerance = tolerance

        AbstractClassSimulation.__init__(self, metos3dModel, parameterId=parameterId, timestep=timestep)

        self._logfile = logfile

        logging.info('***Initialization of DecreasingTimestepsSimulation:***\nMetos3dModel: {:s}\nParameterId: {:d}\nTime step: {:d}dt\nConcentrationId: {:d}'.format(self._metos3dModel, self._parameterId, self._timestep, self._concentrationId))
        logging.info('***Time for initialization: {:.6f}s***\n\n'.format(time.time() - startTime))


    def _init_database(self):
        """
        Inits the database connection
        """
        self._database = DecreasingTimestepsDatabase()


    def set_concentrationId(self, concentrationId=None):
        """
        Sets the id of the initial concentration

        Parameters
        ----------
        concentrationId : int or None, default: None
            Id of the initial concentration. If None, uses the id of the
            standard constant initial concentration
        """
        assert concentrationId is None or type(concentrationId) is int and 0 <= concentrationId

        if concentrationId is None:
            self._concentrationId = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
            self._defaultConcentration = True
        else:
            self._concentrationId = concentrationId
            self._defaultConcentration = False

        self._set_simulationId()
        self._set_path()


    def _set_path(self):
        """
        Sets the path to the simulation directory
        """
        self._path = os.path.join(DecreasingTimesteps_Constants.PATH, 'DecreasingTimesteps', self._metos3dModel, 'Parameter_{:0>3d}'.format(self._parameterId), '{:d}dt'.format(self._timestep), 'Years_{:0>5d}'.format(self._yearInterval), 'Tolerance_{:.1e}'.format(self._tolerance))
        if not self._defaultConcentration:
            self._path = os.path.join(self._path, 'InitialConcentration_{:0>3d}'.format(self._concentrationId))


    def _set_simulationId(self):
        """
        Sets the simulationId
        """
        self._simulationId = self._database.get_simulationId(self._metos3dModel, self._parameterId, self._concentrationId, timestep=self._timestep, yearInterval=self._yearInterval, tolerance=self._tolerance)


    def existsMetos3dOutput(self):
        """
        Checks if the output of metos3d already exists
        """
        checkJobOutput = True
        for year in range(self._yearInterval, self._years, self._yearInterval):
            jobOutput = os.path.join(self._path, DecreasingTimesteps_Constants.PATTERN_OUTPUT_FILENAME.format(year))
            checkJobOutput = checkJobOutput and os.path.exists(jobOutput) and os.path.isfile(jobOutput)
            if not checkJobOutput:
                logging.debug('***existsMetos3dOutput***\nMissing job output: {}'.format(jobOutput))
        return checkJobOutput


    def run(self):
        """
        Run the simulation

        Starts the spin up simulation and decreases automatically the time step
        if the reduction of the relative error is small for a given time step.

        Notes
        -----
        Creates the directory of the simulation
        """
        timeStart = time.time()
        os.makedirs(self._path, exist_ok=True)

        self._year = 0
        self._timestepBefore = self._timestep        

        while (self._year < self._years):
            self._startSimulation()
            logging.info('***Before update Timestep***')
            self._updateTimestep()
            gc.collect()

        timeSimulation = time.time()

        #Rename tracer outputs calculated with one step for the first interval
        for year in range(self._trajectoryYear, self._yearInterval+1, self._trajectoryYear):
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]:
                for end in ['', '.info']:
                    os.rename(os.path.join(self._path, 'TracerOnestep', '{}{}.temp'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer), end)), os.path.join(self._path, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer), end)))

        timeOnestep = time.time()
        logging.info('***Time for simulation {:.6f}s and time for onestep {:.6f}s***\n'.format(timeSimulation - timeStart, timeOnestep - timeSimulation))


    def _startSimulation(self):
        """
        Starts the spin up simulation for one interval
        """
        metos3d = Metos3d(self._metos3dModel, self._timestep, self._modelParameter, self._path, modelYears=self._yearInterval, nodes=self._nodes)
        metos3d.setTrajectoryParameter(trajectoryYear=self._trajectoryYear)

        #Set the initial concentration for the spin up
        if (self._year == 0):
            concentration = [float(c) for c in self._database.get_concentration(self._concentrationId)[Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[self._metos3dModel]]]
            self._tracerStart = np.array(concentration).reshape(len(concentration), 1) * np.ones(shape=(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]), Metos3d_Constants.METOS3D_VECTOR_LEN))
            metos3d.setInitialConcentration(concentration)
        else:
            self._tracerStart = self._tracerEnd
            #Start from the calculated concentration of the last spin up
            metos3d.setInputDir(os.path.join(self._path, 'Tracer'))
            metos3d.setInputTracerName([DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(self._year, tracer, self._timestepBefore) for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]])

        if self._spinupTolerance is not None:
            metos3d.setTolerance(self._spinupTolerance)

        #Run the spin up simulation
        metos3d.run()
 
        self._tracerEnd = metos3d.readTracer()

        #Calculate the tracer concentration for the first time step of a model year
        for year in range(self._trajectoryYear, min(self._lastSpinupYear+1, self._years+1, self._yearInterval+1), self._trajectoryYear):
            logging.debug('Onestep:\nYear: {}'.format(year))
            metos3d.setOneStep(oneStepYear=year)
            metos3d.run()
            if self._removeTracer:
                metos3d.removeTracer(oneStepYear=year)

        #Rename files
        self._renameFiles()


    def _renameFiles(self):
        """
        Rename the files of the last spin up simulation
        """
        #Rename job output
        os.rename(os.path.join(self._path, Metos3d_Constants.PATTERN_OUTPUT_FILENAME), os.path.join(self._path, DecreasingTimesteps_Constants.PATTERN_OUTPUT_FILENAME.format(self._year + self._yearInterval)))

        #Rename output tracer files
        for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]:
            for end in ['', '.info']:
                os.rename(os.path.join(self._path, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer), end)), os.path.join(self._path, 'Tracer', '{}{}'.format(DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(self._year + self._yearInterval, tracer, self._timestep), end)))

        #Rename output tracer calculated with one step
        for year in range(self._trajectoryYear, self._yearInterval+1, self._trajectoryYear):
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]:
                for end in ['', '.info']:
                    if self._year == 0:
                        os.rename(os.path.join(self._path, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer), end)), os.path.join(self._path, 'TracerOnestep', '{}{}.temp'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year + self._year, tracer), end)))
                    else:
                        os.rename(os.path.join(self._path, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer), end)), os.path.join(self._path, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year + self._year, tracer), end)))


    def _updateTimestep(self):
        """
        Update the time step for the next spin up simulation

        Calculate the relative error between the tracer concentration at the
        begin and end of the time interval. Decrease the time step, if the
        relative error is less than the tolerance. Otherwise use the same time
        step again.
        If the tracer concentration contains NaN as value, reject the
        calculated tracer concentration and start the spin up for the interval
        again using the decreased time step.
        """
        logging.debug('***_updateTimestep***')
        reduction = self._calculateTracerNormReduction(self._tracerStart - self._tracerEnd) / self._calculateTracerNormReduction(self._tracerEnd)
        year = self._year
        logging.debug('reduction: {}\nyear: {}'.format(reduction, self._year))

        if (not self._checkTracer() and self._timestep > Metos3d_Constants.METOS3D_TIMESTEPS[0]):
            self._removeTracerOutputs()
            self._decreaseTimestep()
        elif (reduction < self._tolerance):
            self._timestepBefore = self._timestep
            self._year += self._yearInterval
            self._decreaseTimestep()
        else:
            self._timestepBefore = self._timestep
            self._year += self._yearInterval

        logging.info('***updateTimestep***\nReduction: {:.4e}\nYear: {:d}/{:d}\nTimestep: {:d}/{:d}'.format(reduction, self._year, year, self._timestep, self._timestepBefore))


    def _calculateTracerNormReduction(self, tracer, norm='2'):
        """
        Calculate the norm of a tracer concentration

        Parameters
        ----------
        tracer : numpy.ndarray
            Numpy array with the tracer concentration
        norm : str
            Type of the norm

        Return
        ------
        float
            Norm of the given tracer
        """
        assert type(tracer) is np.ndarray
        assert norm in DB_Constants.NORM

        #Set weights for the norm
        if norm == '2':
            norm_vec = np.ones(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN))
        elif norm == 'Boxweighted':
            norm_vec = readBoxVolumes(normvol=True)
        else:
            norm_vec = readBoxVolumes()

        normWeight = np.empty(shape=(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]), len(norm_vec)))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            normWeight[i,:] = norm_vec

        #Norm calculation
        return float(np.sqrt(np.sum((tracer)**2 * normWeight)))


    def _checkTracer(self):
        """
        Check the tracer for invalid entries

        Check if the tracer concentration concentrations invalid values like
        NaN.
        """
        check = np.isnan(self._tracerEnd).any()
        logging.debug('***_checkTracer***\nCheck: {}'.format(check))
        return not np.isnan(self._tracerEnd).any()


    def _decreaseTimestep(self):
        """
        Decrease the time step

        Decrease the time step to the next available time step. If the time
        step is already the smallest time step, the time step remains
        unchanged.
        """
        self._timestep = Metos3d_Constants.METOS3D_TIMESTEPS[max(0, Metos3d_Constants.METOS3D_TIMESTEPS.index(self._timestep)-1)]
        logging.debug('***_decreaseTimestep***\nNew time step: {}'.format(self._timestep))


    def _removeTracerOutputs(self):
        """
        Remove the tracer outputs of the last interval
        """
        #Remove job output
        os.remove(os.path.join(os.path.join(self._path, DecreasingTimesteps_Constants.PATTERN_OUTPUT_FILENAME.format(self._year + self._yearInterval))))

        #Remove output tracer files
        for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]:
            for end in ['', '.info']:
                os.remove(os.path.join(os.path.join(self._path, 'Tracer', '{}{}'.format(DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(self._year + self._yearInterval, tracer, self._timestep), end))))

        #Remove output tracer calculated with one step
        for year in range(self._trajectoryYear, self._yearInterval+1, self._trajectoryYear):
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]:
                for end in ['', '.info']:
                    if self._year == 0:
                        os.remove(os.path.join(self._path, 'TracerOnestep', '{}{}.temp'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year + self._year, tracer), end)))
                    else:
                        os.remove(os.path.join(self._path, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year + self._year, tracer), end)))


    def evaluation(self):
        """
        Evaluation of the spin up simulation

        Notes
        -----
        Inserts the values of the spin up norm, the norm of the tracer
        concentration as well as the norm of the concentration difference
        using a reference solution into the database. Moreover, the values of
        the norm over the whole trajectory is calculated and inserted for the
        last model year of the spin up.
        """
        #Insert the spin up values
        if self.existsMetos3dOutput() and not self._checkSpinupTotalityDatabase():
            self._insertSpinup()

        #Insert the used time step values
        if self.existsMetos3dOutput() and not self._checkTimestepsTotalityDatabase():
            self._insertTimesteps()

        #Insert the tracer norm and deviation values
        if self.existsMetos3dOutput() and (not self._checkNormTotalityDatabase() or not self._checkDeviationTotalityDatabase()):
            self._calculateNorm()

        #Insert the norm of the trajectory
        if self._trajectoryFlag and self.existsMetos3dOutput() and not self._checkTrajectoryNormTotalityDatabase():
            self._calculateTrajectoryNorm()


    def _insertSpinup(self):
        """
        Inserts the spin up norm and convergence values into the database

        Reads the spin up norm values from the Metos3d job output and inserts
        these values into the database.

        Notes
        -----
        Insets also an entry for a convergent or divergent calculation in the
        database.
        """
        for iterationYear in range(self._yearInterval, self._years+1, self._yearInterval):
            metos3d = Metos3d(self._metos3dModel, self._timestep, self._modelParameter, self._path, modelYears=self._yearInterval, nodes=self._nodes)
            spinupNorm = metos3d.read_spinup_norm_values(filenameJoboutput=DecreasingTimesteps_Constants.PATTERN_OUTPUT_FILENAME.format(iterationYear))
            spinupNormShape = np.shape(spinupNorm)

            try:
                for i in range(spinupNormShape[0]):
                    year = int(spinupNorm[i,0])
                    tolerance = float(spinupNorm[i,1])
                    norm = float(spinupNorm[i,2]) if spinupNormShape[1] == 3 else None

                    if year == 0 and tolerance == 0.0 and spinupNorm is not None and norm == 0.0:
                        raise ValueError()

                    self._database.insert_spinup(self._simulationId, year + iterationYear - self._yearInterval, tolerance, norm, overwrite=self._overwrite)

                self._database.insert_convergence(self._simulationId, True, overwrite=self._overwrite)
            except (sqlite3.IntegrityError, ValueError):
                logging.error('Inadmissable values for simulationId {:0>4d} and year {:0>4d}\n'.format(self._simulationId, year))
                self._database.insert_convergence(self._simulationId, False, overwrite=self._overwrite)


    def _checkTimestepsTotalityDatabase(self):
        """
        Checks, if the database contains all values of the used time steps

        Returns
        -------
        bool
            True if the number of database entries coincides with the expected
            number of used time steps values
        """
        expectedCount = int(self._years / self._yearInterval)

        return self._database.check_timesteps(self._simulationId, expectedCount)


    def _insertTimesteps(self):
        """
        Inserts the used time steps values into the database

        Reads the used time steps values from the log file and inserts these
        values into the database.
        """
        year = None
        timestep = None
        reduction = None
        accepted = None

        with open(self._logfile, 'r') as f:
            for line in f:
                matchesReduction = re.search(r'^.* - DEBUG - reduction: (\d+.\d+)', line)
                matchesReductionShort = re.search(r'^Reduction: (\d+.\d+e[+-]\d+)$', line)
                matchesYear = re.search(r'^Year: (\d+)/(\d+)$', line)
                matchesTimesteps = re.search(r'^Timestep: (\d+)/(\d+)$', line)

                #Use only one of the reduction outputs in the logfile for each interval
                if matchesReduction:
                    [reductionStr] = matchesReduction.groups()
                    if reduction is None:
                        reduction = float(reductionStr)
                elif matchesReductionShort:
                    [reductionStr] = matchesReductionShort.groups()
                    if reduction is None:
                        reduction = float(reductionStr)
                elif matchesYear:
                    [yearNew, yearStr] = matchesYear.groups()
                    year = int(yearStr)
                    accepted = not (int(yearNew) == int(yearStr))
                elif matchesTimesteps:
                    [timestepNew, timestepUsed] = matchesTimesteps.groups()
                    timestep = int(timestepUsed)

                    #Insert values in the datbase
                    assert year is not None
                    assert timestep is not None
                    assert accepted is not None
                    self._database.insert_timesteps(self._simulationId, year, timestep, reduction, accepted, overwrite=self._overwrite)
                    year = None
                    timestep = None
                    reduction = None
                    accepted = None


    def _checkNormTotalityDatabase(self, timestepReference=1, yearIntervalReference=10000, toleranceReference=0.01):
        """
        Checks, if the database contains all values of the tracer norm

        Parameters
        ----------
        timestepReference : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the reference solution
        yearIntervalReference : int, default: 10000
            Interval of model years to calculate the spin up with the same
            time step for the reference solution
        toleranceReference : float, default: 0.01
            Tolance used for the reference solution    

        Returns
        -------
        bool
            True if the number of database entries coincides with the expected
            number of tracer norm values
        """
        assert timestepReference in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(yearIntervalReference) is int and yearIntervalReference > 0
        assert type(toleranceReference) is float and toleranceReference > 0

        years = self._years
        if self._spinupTolerance is not None:
            lastYear = self._database.get_spinup_year_for_tolerance(self._simulationId, tolerance=self._spinupTolerance)
            if lastYear is not None:
                years = lastYear
        expectedCount = years//self._trajectoryYear   #Entry for the spin up simulation
        expectedCount = expectedCount + (1 if (years%self._trajectoryYear != 0) else 0) #Entry of the lastYear

        checkNorm = True

        for norm in DB_Constants.NORM:
            checkNorm = checkNorm and self._database.check_tracer_norm(self._simulationId, expectedCount, norm=norm)

            #Norm of the differences
            concentrationIdReference = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
            simulationIdReference = self._database.get_simulationId(self._metos3dModel, self._parameterId, concentrationIdReference, timestep=1, yearInterval=10000, tolerance=0.01)
            checkNorm = checkNorm and self._database.check_difference_tracer_norm(self._simulationId, simulationIdReference, expectedCount, norm=norm)

        return checkNorm


    def _checkDeviationTotalityDatabase(self):
        """
        Checks, if the database contains all values of the tracer deviation

        Returns
        -------
        bool
            True if the number of database entries coincides with the expected
            number of tracer norm values
        """
        years = self._years
        if self._spinupTolerance is not None:
            lastYear = self._database.get_spinup_year_for_tolerance(self._simulationId, tolerance=self._spinupTolerance)
            if lastYear is not None:
                years = lastYear
        expectedCount = years//self._trajectoryYear   #Entry for the spin up simulation
        expectedCount = expectedCount + (1 if (years%self._trajectoryYear != 0) else 0) #Entry of the lastYear

        checkDeviation = True
        checkDeviation = checkDeviation and self._database.check_tracer_deviation(self._simulationId, expectedCount)

        #Norm of the differences
        concentrationIdReference = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
        simulationIdReference = self._database.get_simulationId(self._metos3dModel, self._parameterId, concentrationIdReference, timestep=1, yearInterval=10000, tolerance=0.01)
        checkDeviation = checkDeviation and self._database.check_difference_tracer_deviation(self._simulationId, simulationIdReference, expectedCount)

        return checkDeviation


    def _set_calculateNormReferenceSimulationParameter(self, timestepReference=1, yearIntervalReference=10000, toleranceReference=0.01):
        """
        Returns parameter of the norm calculation

        Parameters
        ----------
        timestepReference : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the reference solution
        yearIntervalReference : int, default: 10000
            Interval of model years to calculate the spin up with the same
            time step for the reference solution
        toleranceReference : float, default: 0.01
            Tolance used for the reference solution    

        Returns
        -------
        tuple
            The tuple contains
              - the simulationId of the simulation used as reference
                simulation and
              - path of the directory of the reference simulation
        """
        assert timestepReference in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(yearIntervalReference) is int and yearIntervalReference > 0
        assert type(toleranceReference) is float and toleranceReference > 0

        concentrationIdReference = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
        simulationIdReference = self._database.get_simulationId(self._metos3dModel, self._parameterId, concentrationIdReference, timestep=timestepReference, yearInterval=yearIntervalReference, tolerance=toleranceReference)

        pathReferenceTracer = os.path.join(Timesteps_Constants.PATH, 'Timesteps', self._metos3dModel, 'Parameter_{:0>3d}'.format(self._parameterId), '{:d}dt'.format(timestepReference))

        return (simulationIdReference, pathReferenceTracer)


    def _calculateNorm(self, simulationIdReference=None, pathReferenceTracer=None):
        """
        Calculates the tracer norm values for every tracer output

        Parameters
        ----------
        simulationIdReference : int or None, default: None
            Id of the simulation used as reference. If None, the function
            _set_calculateNormReferenceSimulationParameter is used to set this
            parameter.
        pathReferenceTracer : str or None, default: None
            Path of the reference simulation directory. If None, the function
            _set_calculateNormReferenceSimulationParameter is used to set this
            parameter.
        """
        assert simulationIdReference is None or type(simulationIdReference) is int and 0 <= simulationIdReference
        assert pathReferenceTracer is None or type(pathReferenceTracer) is str
        assert simulationIdReference is None and pathReferenceTracer is None or simulationIdReference is not None and pathReferenceTracer is not None

        #Parameter of the reference simulation
        if simulationIdReference is None or pathReferenceTracer is None:
            simulationIdReference, pathReferenceTracer = self._set_calculateNormReferenceSimulationParameter()
        pathReferenceTracer = os.path.join(pathReferenceTracer, 'Tracer')
        assert os.path.exists(pathReferenceTracer) and os.path.isdir(pathReferenceTracer)
        tracerReference = self._getTracerOutput(pathReferenceTracer, Metos3d_Constants.PATTERN_TRACER_OUTPUT, year=None)
        yearReference = 10000

        #Tracer's directories
        pathMetos3dTracer = os.path.join(self._path, 'Tracer')
        pathMetos3dTracerOneStep = os.path.join(self._path, 'TracerOnestep')
        assert os.path.exists(pathMetos3dTracer) and os.path.isdir(pathMetos3dTracer)
        assert os.path.exists(pathMetos3dTracerOneStep) and os.path.isdir(pathMetos3dTracerOneStep)

        #Read box volumes
        normvol = readBoxVolumes(normvol=True)
        vol = readBoxVolumes()
        euclidean = np.ones(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN))

        normvol_vec = np.empty(shape=(len(normvol), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
        vol_vec = np.empty(shape=(len(vol), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
        euclidean_vec = np.empty(shape=(len(euclidean), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            normvol_vec[:,i] = normvol
            vol_vec[:,i] = vol
            euclidean_vec[:,i] = euclidean
        normWeight = {'2': euclidean_vec, 'Boxweighted': normvol_vec, 'BoxweightedVol': vol_vec}

        #Insert the tracer norm of the metos3d calculation
        #Initial tracer concentration
        tracerInitialConcentration = self._initialTracerConcentration()

        #Norm for the initial tracer concentration
        for norm in DB_Constants.NORM:
            #Insert the norm values
            self._calculateTracerNorm(tracerInitialConcentration, 0, norm, normWeight[norm])
            self._calculateTracerDifferenceNorm(0, simulationIdReference, yearReference, tracerInitialConcentration, tracerReference, norm, normWeight[norm])

        self._calculateTracerDeviation(tracerInitialConcentration, 0)
        self._calculateTracerDifferenceDeviation(0, simulationIdReference, yearReference, tracerInitialConcentration, tracerReference)

        #Norm of the tracer concentrations during the spin up
        for year in range(self._trajectoryYear, self._years+1, self._trajectoryYear):
            if (year % self._yearInterval) == 0:
                tracer = Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel][0]
                i = 0
                while (i < len(Metos3d_Constants.METOS3D_TIMESTEPS) and (not os.path.exists(os.path.join(pathMetos3dTracer, DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(year, tracer, Metos3d_Constants.METOS3D_TIMESTEPS[i]))))):
                    i += 1
                tracerMetos3dYear = self._getTracerOutput(pathMetos3dTracer, DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP, year=year, timestep=Metos3d_Constants.METOS3D_TIMESTEPS[i])
            else:
                tracerMetos3dYear = self._getTracerOutput(pathMetos3dTracerOneStep, Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR, year=year)

            for norm in DB_Constants.NORM:
                #Insert the norm values
                self._calculateTracerNorm(tracerMetos3dYear, year, norm, normWeight[norm])
                self._calculateTracerDifferenceNorm(year, simulationIdReference, yearReference, tracerMetos3dYear, tracerReference, norm, normWeight[norm])

            self._calculateTracerDeviation(tracerMetos3dYear, year)
            self._calculateTracerDifferenceDeviation(year, simulationIdReference, yearReference, tracerMetos3dYear, tracerReference)


    def _checkTrajectoryNormTotalityDatabase(self, timestepReference=1, yearIntervalReference=10000, toleranceReference=0.01):
        """
        Checks, if the database contains values of the tracer trajectory norm

        Parameters
        ----------
        timestepReference : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the reference solution
        yearIntervalReference : int, default: 10000
            Interval of model years to calculate the spin up with the same
            time step for the reference solution
        toleranceReference : float, default: 0.01
            Tolance used for the reference solution    

        Returns
        -------
        bool
            True if the number of database entries coincides with the expected
            number of tracer norm values
        """
        assert timestepReference in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(yearIntervalReference) is int and yearIntervalReference > 0
        assert type(toleranceReference) is float and toleranceReference > 0

        expectedCount = 1

        checkNorm = True
        for norm in DB_Constants.NORM:
            checkNorm = checkNorm and self._database.check_tracer_norm(self._simulationId, expectedCount, norm=norm, trajectory='Trajectory')

            #Norm of the differences
            concentrationIdReference = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
            simulationIdReference = self._database.get_simulationId(self._metos3dModel, self._parameterId, concentrationIdReference, timestep=timestepReference, yearInterval=yearIntervalReference, tolerance=toleranceReference)
            checkNorm = checkNorm and self._database.check_difference_tracer_norm(self._simulationId, simulationIdReference, expectedCount, norm=norm, trajectory='Trajectory')

        return checkNorm


    def _calculateTrajectoryNorm(self):
        """
        Calculates the trajectory norm

        Notes
        -----
        The trajectory is always computed with time step 1dt.
        """
        timestep = 1
        #Parameter of the reference simulation
        simulationIdReference, referenceTrajectoryPath = self._set_calculateNormReferenceSimulationParameter()
        referenceTrajectoryPath = os.path.join(referenceTrajectoryPath, 'Trajectory')
        os.makedirs(referenceTrajectoryPath, exist_ok=True)
        yearReference = 10000
        trajectoryReference = self._calculateTrajectory(referenceTrajectoryPath, year=yearReference, timestep=timestep, reference=True)

        #Read box volumes
        normvol = readBoxVolumes(normvol=True)
        vol = readBoxVolumes()
        euclidean = np.ones(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN))

        normvol_vec = np.empty(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        vol_vec = np.empty(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        euclidean_vec = np.empty(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        for t in range(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep)):
            for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
                normvol_vec[t,i,:] = normvol
                vol_vec[t,i,:] = vol
                euclidean_vec[t,i,:] = euclidean
        normWeight = {'2': euclidean_vec, 'Boxweighted': normvol_vec, 'BoxweightedVol': vol_vec}

        #Insert trajectory norm values of the reference simulation
        for norm in DB_Constants.NORM:
            if not self._database.check_tracer_norm(simulationIdReference, 1, norm=norm, trajectory='Trajectory'):
                self._calculateTrajectoryTracerNorm(trajectoryReference, yearReference, norm, normWeight[norm], timestep=timestep, simulationId=simulationIdReference)

        #Trajectory of the simulation
        trajectoryPath = os.path.join(self._path, 'Trajectory')
        os.makedirs(trajectoryPath, exist_ok=True)

        lastYear = self._years
        if self._spinupTolerance is not None:
            metos3d = Metos3d(self._metos3dModel, timestep, self._modelParameter, self._path, modelYears=self._years, nodes=self._nodes)
            lastYear = metos3d.lastSpinupYear()

        #Read trajectory
        trajectory = self._calculateTrajectory(trajectoryPath, year=self._years, timestep=timestep)

        for norm in DB_Constants.NORM:
            self._calculateTrajectoryTracerNorm(trajectory, self._years, norm, normWeight[norm], timestep=timestep)
            self._calculateTrajectoryDifferenceTracerNorm(self._years, simulationIdReference, yearReference, trajectory, trajectoryReference, norm, normWeight[norm], timestep=timestep)

        #Remove the directory for the trajectory
        shutil.rmtree(trajectoryPath, ignore_errors=True)


    def _calculateTrajectory(self, metos3dSimulationPath, year=10000, timestep=1, modelYears=0, reference=False):
        """
        Calculates the trajectory

        Parameters
        ----------
        metos3dSimulationPath : str
            Path for the simulation with Metos3d
        year: int, default: 10000
            Model year of the spin up simulation for the trajectory
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation to calculate the trajectory
        modelYears : int
            Model years for Metos3d
        reference : bool, default: False
            If True, use the path of the reference simulation to copy the
            tracer concentration used as initial concentration for the
            trajectory

        Returns
        -------
        numpy.ndarray
            Numpy array with the trajectory
        """
        assert os.path.exists(metos3dSimulationPath) and os.path.isdir(metos3dSimulationPath)
        assert year in range(0, self._years+1)
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(modelYears) is int and 0 <= modelYears
        assert type(reference) is bool

        #Run metos3d
        tracer_path = os.path.join(metos3dSimulationPath, 'Tracer')
        os.makedirs(tracer_path, exist_ok=True)

        metos3d = Metos3d(self._metos3dModel, timestep, self._modelParameter, metos3dSimulationPath, modelYears=modelYears, nodes=self._nodes)
        metos3d.setCalculateOnlyTrajectory()

        if not self._existsTrajectory(tracer_path, timestep=timestep):
            #Copy the input tracer for the trajectory
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]:
                if year == 10000:
                    if reference:
                        inputTracer = os.path.join(os.path.dirname(metos3dSimulationPath), 'Tracer', Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer))
                    else:
                        i = 0
                        while (i < len(Metos3d_Constants.METOS3D_TIMESTEPS) and (not os.path.exists(os.path.join(self._path, 'Tracer', DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(year, tracer, Metos3d_Constants.METOS3D_TIMESTEPS[i]))))):
                            i += 1
                        inputTracer = os.path.join(self._path, 'Tracer', DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(year, tracer, Metos3d_Constants.METOS3D_TIMESTEPS[i]))
                else:
                    inputTracer = os.path.join(os.path.dirname(metos3dSimulationPath) if reference else self._path, 'TracerOnestep', Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer))
                shutil.copy(inputTracer, os.path.join(tracer_path, Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer)))

            metos3d.run()

        #Read tracer concentration
        trajectory = metos3d.readTracer()

        return trajectory

