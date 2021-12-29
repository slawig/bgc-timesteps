#!/usr/bin/env python
# -*- coding: utf8 -*

import logging
import numpy as np
import os
import sqlite3
import time

import metos3dutil.latinHypercubeSample.constants as LHS_Constants
import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.database.DatabaseMetos3d import DatabaseMetos3d
import stepSizeControl.constants as SSC_Constants


class StepSizeControlDatabase(DatabaseMetos3d):
    """
    Access functions for the database
    """

    def __init__(self, dbpath=SSC_Constants.DB_PATH, completeTable=True, createDb=False):
        """
        Initialization of the database connection

        Parameter
        ----------
        dbpath : str, default: stepSizeControl.constants.DB_PATH
            Path to the sqlite database
        completeTable : bool, default: True
            If the value is True, use all columns (even columns with value
            None) in SELECT queries on the database
        createDb : bool, default: False
            If True, the database does not have to exist and can be created
            using the function create_database

        Raises
        ------
        AssertionError
            If the file for the sqlite database does not exist
        """
        assert type(createDb) is bool
        assert createDb or os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        DatabaseMetos3d.__init__(self, dbpath, completeTable=completeTable, createDb=createDb)


    def create_database(self):
        """
        Create all tables of the database
        """
        self._c.execute('PRAGMA foreign_keys=on')
        self._create_table_parameter()
        self._create_table_initialConcentration()
        self._create_table_simulation()
        self._create_table_spinup()
        self._create_table_stepControl()
        self._create_table_timeStepControl()
        self._create_table_timeStepControlOneYear()
        self._create_table_time()

        #Database tables for the norm
        for norm in DB_Constants.NORM:
            self._create_table_tracerNorm(norm=norm)
            self._create_table_tracerNorm(norm=norm, trajectory='Trajectory')
            self._create_table_tracerDifferenceNorm(norm=norm)
            self._create_table_tracerDifferenceNorm(norm=norm, trajectory='Trajectory')
            self._create_table_tracerDifferenceLhsNorm(norm=norm)

        #TODO Change TracerDifference*Norm tables: Structure with two simulationIds and years (remove *_Endpoint columns)
        #TODO Rename Tracer*NormTrajetory tables in TracerTrajecory*Norm
        #TODO Create Difference Lhs tables for each norm

        #Database tables for the deviation
        self._create_table_DeviationTracer()
        self._create_table_DeviationTracerDifference()

        self._conn.commit()

        #Initial insert of data sets
        self._init_database()


    def _init_database(self):
        """
        Initial insert of the database tables

        Notes
        -----
        The functions inserts data sets into the tables Parameter,
        InitialConcentration and Simulation
        """
        #Insert the reference parameter set and the parameter of the latin hypercube sample with 100 samples into the table Parameter
        self._init_table_parameter(referenceParameter=True, latinHypercubeSamples=(True, False, False))

        #Insert the constant initial concentration
        self._init_table_initialConcentration()

        #Insert the simulation data sets
        self._init_table_simulation()


    def _create_table_simulation(self):
        """
        Create table Simulation
        """
        self._c.execute('''CREATE TABLE Simulation (simulationId INTEGER NOT NULL, model TEXT NOT NULL, parameterId INTEGER NOT NULL REFERENCES Parameter(parameterId), concentrationId INTEGER NOT NULL REFERENCES InitialConcentration(concentrationId), startTimestep INTEGER NOT NULL, stepYear INTEGER NOT NULL, tolerance REAL NOT NULL, rho REAL NOT NULL, eta REAL NOT NULL, cpus INTEGER NOT NULL, norm TEXT NOT NULL, checkConcentration INTEGER NOT NULL DEFAULT 0, singleStep INTEGER NOT NULL DEFAULT 0, singleStepYear INTEGER NOT NULL DEFAULT 0, upperLayer INTEGER NOT NULL DEFAULT 0, lowerLayer INTEGER NOT NULL DEFAULT 14, UNIQUE (model, parameterId, concentrationId, startTimestep, stepYear, tolerance, rho, eta, cpus, norm, checkConcentration, singleStep, singleStepYear, upperLayer, lowerLayer), PRIMARY KEY (simulationId))''')


    def _init_table_simulation(self):
        """
        Initial insert of simulation data sets
        """
        #TODO Implement this initialisation of the simulation database table
        pass


    def exists_simulation(self, metos3dModel, parameterId, concentrationId, startTimestep=1, stepYear=1, tolerance=1.0, rho=1.0, eta=2.0, cpus=64, norm='BoxweightedVol', checkConcentration=False, singleStep=False, singleStepYear=False, upperLayer=None, lowerLayer=None):
        """
        Returns if a simulation entry exists for the given values

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        concentrationId : int
            Id of the concentration
        startTimestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Initial time step for the step size control
        tolerance : float, default: 1.0
            Tolerance for the local discretization error estimation
        rho : float, default: 1.0
            Factor for the tolerance used in the calculation of the
            optimial step size
        eta : float, default: 1.0
            Factor for scaling the step size for the next step
        cpus : int, default: 64
            Number of used cpus
        norm : {'2', 'BoxweightedVol', 'Boxweighted'}, default: 'BoxweightedVol'
            Used norm to estimate the local discretization error
        checkConcentration : bool, default: False
            If True, accept only concentrations with not negative
            concentrations
        singleYear : bool, default: False
            If True, use a single step (instead of model years) for the step
            size control
        singleStepYear : bool, default: False
            If True, the step size control used at the end of each model year
            the step size to calculate the concentration for the first time
            instant in the next model year. Otherwise, it is possible that the
            step size control does not compute the concentration at the first
            time instant.
        upperLayer : None or int, default: None
            If the value is an int, restrict the norm to the layer in ocean
            using the upperLayer index 'upperLayer'.
        lowerLayer : None or int, default: None
            If the value is an int, restrict the norm to the layer in ocean
            using the lowerLayer index 'lowerLayer'.

        Returns
        -------
        bool
            True if an entry exists for the given values
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and 0 <= concentrationId
        assert startTimestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(tolerance) is float and 0.0 < tolerance
        assert type(rho) is float and 0.0 < rho
        assert type(eta) is float and 0.0 < eta
        assert type(cpus) is int and 0 < cpus
        assert norm in DB_Constants.NORM
        assert type(checkConcentration) is bool
        assert type(singleYear) is bool
        assert type(singleStepYear) is bool
        assert upperLayer is None or type(upperLayer) is int and 0 <= upperLayer
        assert lowerLayer is None or type(lowerLayer) is int and 0 <= lowerLayer

        upperLayerStr = ' AND upperLayer IS NULL' if upperLayer is None else ' AND upperLayer = {:d}'.format(upperLayer)
        lowerLayerStr = ' AND lowerLayer IS NULL' if lowerLayer is None else ' AND lowerLayer = {:d}'.format(lowerLayer)

        sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND parameterId = ? AND concentrationId = ? AND startTimestep = ? AND stepYear = ? AND tolerance = ? AND rho = ? AND eta = ? AND cpus = ? AND norm = ? AND checkConcentration = ? AND singleStep = ? AND singleStepYear = ?{}{}'.format(upperLayerStr, lowerLayerStr)
        self._c.execute(sqlcommand, (metos3dModel, parameterId, concentrationId, startTimestep, stepYear, tolerance, rho, eta, cpus, norm, int(checkConcentration), int(singleStep), int(singleStepYear)))
        simulationId = self._c.fetchall()
        return len(simulationId) > 0


    def get_simulationId(self, metos3dModel, parameterId, concentrationId, startTimestep=1, stepYear=1, tolerance=1.0, rho=1.0, eta=2.0, cpus=64, norm='BoxweightedVol', checkConcentration=False, singleStep=False, singleStepYear=False, upperLayer=None, lowerLayer=None):
        """
        Returns the simulationId

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        concentrationId : int
            Id of the concentration
        startTimestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Initial time step for the step size control
        stepYear : int, default: 1
            Number of model years without adapting the time step
        tolerance : float, default: 1.0
            Tolerance for the local discretization error estimation
        rho : float, default: 1.0
            Factor for the tolerance used in the calculation of the
            optimial step size
        eta : float, default: 1.0
            Factor for scaling the step size for the next step
        cpus : int, default: 64
            Number of used cpus
        norm : {'2', 'BoxweightedVol', 'Boxweighted'}, default: 'BoxweightedVol'
            Used norm to estimate the local discretization error
        checkConcentration : bool, default: False
            If True, accept only concentrations with not negative
            concentrations
        singleYear : bool, default: False
            If True, use a single step (instead of model years) for the step
            size control
        singleStepYear : bool, default: False
            If True, the step size control used at the end of each model year
            the step size to calculate the concentration for the first time
            instant in the next model year. Otherwise, it is possible that the
            step size control does not compute the concentration at the first
            time instant.
        upperLayer : None or int, default: None
            If the value is an int, restrict the norm to the layer in ocean
            using the upperLayer index 'upperLayer'.
        lowerLayer : None or int, default: None
            If the value is an int, restrict the norm to the layer in ocean
            using the lowerLayer index 'lowerLayer'.

        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation
        yearInterval : int, default: 50
            Number of model years of each spin up
        tolerance : float, default: 0.01
            Boarder for the error to decrease the time stepp

        Returns
        -------
        int
            simulationId for the combination of model, parameterId,
            concentrationId, startTimestep, stepYear, tolerance, rho, eta,
            cpus, norm, checkConcentration, singleYear, singleStepYear,
            upperLayer and lowerLayer

        Raises
        ------
        AssertionError
            If no entry exists for this combination in the database
            table Simulation
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and 0 <= concentrationId
        assert startTimestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(stepYear) is int and 0 < stepYear
        assert type(tolerance) is float and 0.0 < tolerance
        assert type(rho) is float and 0.0 < rho
        assert type(eta) is float and 0.0 < eta
        assert type(cpus) is int and 0 < cpus
        assert norm in DB_Constants.NORM
        assert type(checkConcentration) is bool
        assert type(singleYear) is bool
        assert type(singleStepYear) is bool
        assert upperLayer is None or type(upperLayer) is int and 0 <= upperLayer
        assert lowerLayer is None or type(lowerLayer) is int and 0 <= lowerLayer

        upperLayerStr = ' AND upperLayer IS NULL' if upperLayer is None else ' AND upperLayer = {:d}'.format(upperLayer)
        lowerLayerStr = ' AND lowerLayer IS NULL' if lowerLayer is None else ' AND lowerLayer = {:d}'.format(lowerLayer)

        sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND parameterId = ? AND concentrationId = ? AND startTimestep = ? AND stepYear = ? AND tolerance = ? AND rho = ? AND eta = ? AND cpus = ? AND norm = ? AND checkConcentration = ? AND singleStep = ? AND singleStepYear = ?{}{}'.format(upperLayerStr, lowerLayerStr)
        self._c.execute(sqlcommand, (metos3dModel, parameterId, concentrationId, startTimestep, stepYear, tolerance, rho, eta, cpus, norm, int(checkConcentration), int(singleStep), int(singleStepYear)))
        simulationId = self._c.fetchall()
        assert len(simulationId) == 1
        return simulationId[0][0]


    def insert_simulation(self, metos3dModel, parameterId, concentrationId, startTimestep=1, stepYear=1, tolerance=1.0, rho=1.0, eta=2.0, cpus=64, norm='BoxweightedVol', checkConcentration=False, singleStep=False, singleStepYear=False, upperLayer=None, lowerLayer=None):
        """
        Insert simulation data set

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        concentrationId : int
            Id of the concentration
        startTimestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Initial time step for the step size control
        stepYear : int, default: 1
            Number of model years without adapting the time step
        tolerance : float, default: 1.0
            Tolerance for the local discretization error estimation
        rho : float, default: 1.0
            Factor for the tolerance used in the calculation of the
            optimial step size
        eta : float, default: 1.0
            Factor for scaling the step size for the next step
        cpus : int, default: 64
            Number of used cpus
        norm : {'2', 'BoxweightedVol', 'Boxweighted'}, default: 'BoxweightedVol'
            Used norm to estimate the local discretization error
        checkConcentration : bool, default: False
            If True, accept only concentrations with not negative
            concentrations
        singleYear : bool, default: False
            If True, use a single step (instead of model years) for the step
            size control
        singleStepYear : bool, default: False
            If True, the step size control used at the end of each model year
            the step size to calculate the concentration for the first time
            instant in the next model year. Otherwise, it is possible that the
            step size control does not compute the concentration at the first
            time instant.
        upperLayer : None or int, default: None
            If the value is an int, restrict the norm to the layer in ocean
            using the upperLayer index 'upperLayer'.
        lowerLayer : None or int, default: None
            If the value is an int, restrict the norm to the layer in ocean
            using the lowerLayer index 'lowerLayer'.

        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation
        yearInterval : int, default: 50
            Number of model years of each spin up
        tolerance : float, default: 0.01
            Boarder for the error to decrease the time stepp

        Returns
        -------
        int
            simulationId for the combination of model, parameterId,
            concentrationId, startTimestep, stepYear, tolerance, rho, eta,
            cpus, norm, checkConcentration, singleYear, singleStepYear,
            upperLayer and lowerLayer

        Raises
        ------
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and 0 <= concentrationId
        assert startTimestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(stepYear) is int and 0 < stepYear
        assert type(tolerance) is float and 0.0 < tolerance
        assert type(rho) is float and 0.0 < rho
        assert type(eta) is float and 0.0 < eta
        assert type(cpus) is int and 0 < cpus
        assert norm in DB_Constants.NORM
        assert type(checkConcentration) is bool
        assert type(singleYear) is bool
        assert type(singleStepYear) is bool
        assert upperLayer is None or type(upperLayer) is int and 0 <= upperLayer
        assert lowerLayer is None or type(lowerLayer) is int and 0 <= lowerLayer

        if self.exists_simulation(metos3dModel, parameterId, concentrationId, startTimestep=startTimestep, stepYear=stepYear, tolerance=tolerance, rho=rho, eta=eta, cpus=cpus, norm=norm, checkConcentration=checkConcentration, singleStep=singleStep, singleStepYear=singleStepYear, upperLayer=upperLayer, lowerLayer=lowerLayer):
            #Simulation already exists in the database
            simulationId = self.get_simulationId(metos3dModel, parameterId, concentrationId, startTimestep=startTimestep, stepYear=stepYear, tolerance=tolerance, rho=rho, eta=eta, cpus=cpus, norm=norm, checkConcentration=checkConcentration, singleStep=singleStep, singleStepYear=singleStepYear, upperLayer=upperLayer, lowerLayer=lowerLayer)
        else:
            #Insert simulation into the database
            sqlcommand = 'SELECT MAX(simulationId) FROM Simulation'
            self._c.execute(sqlcommand)
            dataset = self._c.fetchall()
            assert len(dataset) == 1
            simulationId = dataset[0][0] + 1

            purchases = [(simulationId, metos3dModel, parameterId, concentrationId, startTimestep, stepYear, tolerance, rho, eta, cpus, norm, int(checkConcentration), int(singleStep), int(singleStepYear))]
            inserted = False
            insertCount = 0
            while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
                try:
                    self._c.executemany('INSERT INTO Simulation VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', purchases)
                    self._conn.commit()
                    inserted = True
                except sqlite3.OperationalError:
                    insertCount += 1
                    #Wait for the next insert
                    time.sleep(DB_Constants.TIME_SLEEP)

        return simulationId


    def _create_table_spinup(self):
        """
        Create table Spinup
        """
        self._c.execute('''CREATE TABLE Spinup (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, timestepSmall INTEGER NOT NULL, timestepBig INTEGER NOT NULL, toleranceTimestepSmall REAL, spinupNormTimestepSmall REAL, toleranceTimestepBig REAL, spinupNormTimestepBig REAL, PRIMARY KEY (simulationId, year, timestepSmall))''')


    def get_spinup_year_for_tolerance(self, simulationId, tolerance=0.0001):
        """
        Returns the first model year of the spin up with less tolerance

        Returns the model year of the spin up calculation where the tolerance
        fall below the given tolerance value. If the tolerance of the spin up
        is higher than the given tolerance for every model year, return None.

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        tolerance : float, default: 0.0001
            Tolerance value for the spin up norm

        Returns
        -------
        None or int
            If the spin up norm is always greater than the given tolerance,
            return None. Otherwise, the model year in which the spin up norm
            falls below the tolerance for the first time is returned.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(tolerance) is float and tolerance > 0

        sqlcommand = 'SELECT sp.year FROM Spinup AS sp WHERE sp.simulationId = ? AND sp.toleranceTimestepSmall < ? AND NOT EXISTS (SELECT * FROM Spinup AS sp1 WHERE sp1.simulationId = sp.simulationId AND sp1.toleranceTimestepSmall < ? AND sp1.year < sp.year)'
        self._c.execute(sqlcommand, (simulationId, tolerance, tolerance))
        count = self._c.fetchall()
        assert len(count) == 1 or len(count) == 0
        if len(count) == 1:
            return count[0][0] + 1
        else:
            return None


    def read_spinup_values_for_simid(self, simulationId, reference=False):
        """
        Returns the spin up norm values of a simulation

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        reference : bool, default: False
            If True, the simulationId represents the reference solution

        Returns
        -------
        numpy.ndarray
            2D array with the year and the tolerance used the small and the
            big time step
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(reference) is bool

        if reference:
            sqlcommand = 'SELECT sp.year, sp.toleranceTimestepSmall, sp.toleranceTimestepBig FROM Spinup AS sp WHERE sp.simulationId = ? ORDER BY sp.year;'
            self._c.execute(sqlcommand, (simulationId, ))
        else:
            sqlcommand = 'SELECT sp.year, sp.toleranceTimestepSmall, sp.toleranceTimestepBig FROM Spinup AS sp, StepControl AS sc WHERE sp.simulationId = ? AND sp.simulationId = sc.simulationId AND sp.year = sc.year AND sp.timestepSmall = sc.timestepSmall AND sc.accept = ? ORDER BY sp.year;'
            self._c.execute(sqlcommand, (simulationId, int(True)))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 3))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1], row[2]])
            i = i+1
        return simdata


    def read_spinup_tolerance(self, metos3dModel, concentrationId, year):
        """
        Returns the spin up tolerance for all parameterIds

        Returns the spin up tolerance of all simulations using the given model
        and concentrationId for the given model year.

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        concentrationId : int
            Id of the concentration
        year : int
            Model year of the spin up calculation

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId and the tolerance
        """
        pass


    def read_spinup_year(self, model, concentrationId):
        """
        Returns the required years to reach the given spin up tolerance

        Returns the required model years to reach the given spin up tolerance
        for every parameterId.

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        concentrationId : int
            Id of the concentration

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId and the required model year
        """
        pass


    def insert_spinup(self, simulationId, year, timestepSmall, timestepBig, toleranceTimestepSmall, spinupNormTimestepSmall, toleranceTimestepBig, spinupNormTimestepBig, overwrite=False):
        """
        Insert spin up value

        Insert spin up value. If a spin up database entry for the simulationId
        and year already exists, the existing entry is deleted and the new one
        is inserted (if the flag overwrite is True).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year of the spin up calculation
        timestepSmall : {1, 2, 4, 8, 16, 32, 64}
            Small time step used to estimate the local discretization error
        timestepBig : {1, 2, 4, 8, 16, 32, 64}
            Big time step used to estimate the local discretization error
        toleranceTimestepSmall : float
            Tolerance of the spin up norm used the small time step
        spinupNormTimestepSmall : float
            Spin up Norm value used the small time step
        toleranceTimestepBig : float
            Tolerance of the spin up norm used the big time step
        spinupNormTimestepBig : float
            Spin up Norm value used the big time step
        overwrite : bool, default: False
            If True, overwrite an existing entry.

        Raises
        ------
        AssertionError
            If an existing entry exists and should not be overwritten.
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts.

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert timestepSmall in Metos3d_Constants.METOS3D_TIMESTEPS
        assert timestepBig in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(toleranceTimestepSmall) is float and 0 <= toleranceTimestepSmall
        assert type(spinupNormTimestepSmall) is float and 0 <= spinupNormTimestepSmall
        assert type(toleranceTimestepBig) is float and 0 <= toleranceTimestepBig
        assert type(spinupNormTimestepBig) is float and 0 <= spinupNormTimestepBig
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationId, year FROM Spinup WHERE simulationId = ? AND year = ? AND timestepSmall = ?'
        self._c.execute(sqlcommand_select, (simulationId, year, timestepSmall))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Spinup WHERE simulationId = ? AND year = ? AND timestepSmall'
                self._c.execute(sqlcommand, (simulationId, year, timestepSmall))
        else:
            assert len(dataset) == 0

        #Generate and insert spin-up value
        purchases = []
        purchases.append((simulationId, year, timestepSmall, timestepBig, toleranceTimestepSmall, spinupNormTimestepSmall, toleranceTimestepBig, spinupNormTimestepBig))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO Spinup VALUES (?,?,?,?,?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def _create_table_stepControl(self):
        """
        Create table stepControl
        """
        self._c.execute('''CREATE TABLE StepControl (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, timestepSmall INTEGER NOT NULL, timestepBig INTEGER NOT NULL, accept INTEGER NOT NULL, dropCount INTEGER NOT NULL, differenceNorm REAL, estimatedError REAL, hbest REAL, miminum REAL, hmin REAL, PRIMARY KEY (simulationId, year, timestepSmall, dropCount))''')


    def insert_stepcontrol(self, simulationId, year, timestepSmall, timestepBig, accept, dropCount, differenceNorm, estimatedError, hbest, minimum, hmin, overwrite=False):
        """
        Insert values of one step control step

        Insert step control value. If a step control database entry for the
        simulationId, year timestepSmall and dropCount already exists, the
        existing entry is deleted and the new one is inserted (if the flag
        overwrite is True).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year of the spin up calculation
        timestepSmall : {1, 2, 4, 8, 16, 32, 64}
            Small time step used to estimate the local discretization error
        timestepBig : {1, 2, 4, 8, 16, 32, 64}
            Big time step used to estimate the local discretization error
        accept : bool
            If True, the step of the step size control is accepted
        dropCount : int
            Number of iterations without accepting the step
        differenceNorm : float
            Norm of the difference used to estimate the local discretization
            error
        estimatedError : float
            Value of the estimated local discretization error
        hbest : float
            Optimal step size
        minimum : float
            Value for the next step size
        hmin : float
            Minimal step size
        overwrite : bool, default: False
            If True, overwrite an existing entry.

        Raises
        ------
        AssertionError
            If an existing entry exists and should not be overwritten.
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts.

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert timestepSmall in Metos3d_Constants.METOS3D_TIMESTEPS
        assert timestepBig in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(accept) is bool
        assert type(dropCount) is int and 0 <= dropCount
        assert type(differenceNorm) is float and 0.0 <= differenceNorm
        assert type(estimatedError) is float and 0.0 <= estimatedError
        assert type(hbest) is float and 0.0 <= hbest
        assert type(minimum) is float and 0.0 <= minumum
        assert type(hmin) is float and 0.0 <= hmin
        assert type(overwrite) is bool

        sqlcommand = 'SELECT simulationId, year, timestepSmall, dropCount FROM StepControl WHERE simulationId = ? AND year = ? and timestepSmall = ? AND dropCount = ?'
        self._c.execute(sqlcommand, (simulationId, year, timestepSmall, dropCount))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM StepControl WHERE simulationId = ? AND year = ? AND timestepSmall = ? AND dropCount = ?'
                self._c.execute(sqlcommand, (simulationId, year, timestepSmall, dropCount))
        else:
            assert len(dataset) == 0

        #Generate and insert step control value
        purchases = []
        purchases.append((simulationId, year, timestepSmall, timestepBig, int(accept), dropCount, differenceNorm, estimatedError, hbest, minimum, hmin))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO StepControl VALUES (?,?,?,?,?,?,?,?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def get_timestep_stepcontrol(self, simulationId, year):
        """
        Returns the used time step of the step size control

        Returns the used time step of the step size control
        for the given simulationId and year

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year of the spin up calculation

        Returns
        -------
        int
            Time step of the step size control for the given year
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0

        sqlcommand = 'SELECT timestepSmall, dropCount FROM StepControl WHERE simulationId = ? AND year = ? AND accept = ?'
        self._c.execute(sqlcommand,  (simulationId, year, int(True)))
        timestep = self._c.fetchall()
        assert len(timestep) == 1
        return timestep[0][0]


    def checkStepControl(self, simulationId, exceptedCount):
        """
        Check the number of step control entries for the given simulationId

        Check the number of entries in the table StepControl for with the given
        simulationId. We use only the accepted approximations entries and omit
        the entries of dropped approximations.

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        expectedCount : int
            Expected number of database entries for the spin up

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(expectedCount) is int and expectedCount >= 0

        sqlcommand = 'SELECT COUNT(*) FROM StepControl WHERE simulationId = ? AND accept = ?'
        self._c.execute(sqlcommand, (simulationId, int(True)))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***CheckStepControl: Expected: {} Get: {}***'.format(expectedCount, count[0][0]))
        return count[0][0] == exceptedCount


    def read_stepControl_values_for_simulationId(self, simulationId):
        """
        Returns values of the used steps for a given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        numpy.ndarray
            Array including the year, timestepSmall and timestepBig used for
            the step size control over the whole spin up
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT year, timestepSmall, timestepBig, accept FROM StepControl WHERE simulationId = ? ORDER BY year;'
        self._c.execute(sqlcommand,  (simulationId, ))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 4))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1], row[2], row[3]])
            i = i+1

        return simdata


    def _create_table_time(self):
        """
        Create table time
        """
        self._c.execute('''CREATE TABLE Time (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), timeInitialisation REAL NOT NULL, overallTime REAL NOT NULL, PRIMARY KEY (simulationId))''')


    def insert_time(self, simulationId, timeInitialisation, overallTime, overwrite=False):
        """
        Insert values in the time table

        Insert time value. If a time database entry for the simulationId
        already exists, the  existing entry is deleted and the new one is
        inserted (if the flag overwrite is True).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        timeInitialisation : float
            Time required to initial the step size control run
        overallTime : float
            Overall time of the step size control run
        overwrite : bool, default: False
            If True, overwrite an existing entry.

        Raises
        ------
        AssertionError
            If an existing entry exists and should not be overwritten.
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts.

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(timeInitialisation) is float and 0.0 < timeInitialisation
        assert type(overallTime) is float and 0.0 < overallTime
        assert type(overwrite) is bool

        sqlcommand = 'SELECT simulationId FROM Time WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Time WHERE simulationId = ?'
                self._c.execute(sqlcommand, (simulationId, ))
        else:
            assert len(dataset) == 0

        #Generate and insert time value
        purchases = []
        purchases.append((simulationId, timeInitialisation, overallTime))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO time VALUES (?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def checkTime(self, simulationId, expectedCount = 1):
        """
        Check the number of entries in the table Time for the given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        expectedCount : int, default: 1
            Expected number of database entries for the spin up

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(expectedCount) is int and expectedCount >= 0

        sqlcommand = 'SELECT COUNT(*) FROM Time WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        count = self._c.fetchall()
        assert len(count) == 1
        return count[0][0] == 1


    def _create_table_timeStepControl(self):
        """
        Create table timeStepControl
        """
        self._c.execute('''CREATE TABLE TimeStepControl (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, timestep INTEGER NOT NULL, dropCount INTEGER NOT NULL, overallTime REAL NOT NULL, timeSetTimestep REAL NOT NULL, timeMetos3d REAL NOT NULL, timeErrorEstimation REAL NOT NULL, timeTimestepAcception REAL NOT NULL, timeOptimalTimestep REAL, timeSetNewTimestep REAL, PRIMARY KEY (simulationId, year, timestep, dropCount))''')


    def insert_timeStepControl(self, simulationId, year, timestep, dropCount, overallTime, timeSetTimestep, timeMetos3d, timeErrorEstimation, timeTimestepAcception, timeOptimalTimestep, timeSetNewTimestep, overwrite=False):
        """
        Insert values in the time step control table

        Insert time step control value. If a time step control database entry
        for the simulationId, year, timestep and dropCount already exists, the
        existing entry is deleted and the new one is inserted (if the flag
        overwrite is True).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year of the spin up calculation
        timestep : {1, 2, 4, 8, 16, 32, 64}
            Time step used to estimate the local discretization error
        dropCount : int
            Number of iterations without accepting the step
        overallTime : float
            Overall time of the step size control step
        timeSetTimestep : float
            Time to set the time step
        timeMetos3d : float
            Time to run Metos3d (spin-up simulation)
        timeErrorEstimation : float
            Time to estimate the local discretization error
        timeTimestepAcception : float
            Required time to compute the acception of the time step
        timeOptimalTimestep : float
            Required time to compute the optimal time step
        timeSetNewTimestep : float
            Time setting the new time step
        overwrite : bool, default: False
            If True, overwrite an existing entry.

        Raises
        ------
        AssertionError
            If an existing entry exists and should not be overwritten.
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts.

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(dropCount) is int and 0 <= dropCount
        assert type(overallTime) is float and 0.0 < overallTime
        assert type(timeSetTimestep) is float and 0.0 < timeSetTimestep
        assert type(timeMetos3d) is float and 0.0 < timeMetos3d
        assert type(timeErrorEstimation) is float and 0.0 < timeErrorEstimation
        assert type(timeTimestepAcception) is float and 0.0 < timeTimestepAcception
        assert type(timeOptimalTimestep) is float and 0.0 < timeOptimalTimestep
        assert type(timeSetNewTimestep) is float and 0.0 < timeSetNewTimestep
        assert type(overwrite) is bool

        sqlcommand = 'SELECT simulationId, year, timestep, dropCount FROM TimeStepControl WHERE simulationId = ? AND year = ? AND timestep = ? AND dropCount = ?'
        self._c.execute(sqlcommand, (simulationId, year, timestep, dropCount))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM TimeStepControl WHERE simulationId = ? AND year = ? AND timestep = ? AND dropCount = ?'
                self._c.execute(sqlcommand, (simulationId, year, timestep, dropCount))
        else:
            assert len(dataset) == 0

        purchases = []
        purchases.append((simulationId, year, timestep, dropCount, overallTime, timeSetTimestep, timeMetos3d, timeErrorEstimation, timeTimestepAcception, timeOptimalTimestep, timeSetNewTimestep))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO TimeStepControl VALUES (?,?,?,?,?,?,?,?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def checkTimeStepControl(self, simulationId, exceptedCount):
        """
        Check the number of entries in the table TimeStepControl

        Check the number of entries in the table TimeStepControl for the given
        simulationId. We use only the accepted approximations entries and omit
        the entries of dropped approximations.

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        expectedCount : int, default: 1
            Expected number of database entries for the spin up

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(expectedCount) is int and expectedCount >= 0

        sqlcommand = 'SELECT COUNT(*) FROM TimeStepControl AS sc WHERE sc.simulationId = ? AND NOT EXISTS (SELECT * FROM TimeStepControl AS sc1 WHERE sc.simulationId = sc1.simulationId AND sc.year = sc1.year AND sc.dropCount < sc1.dropCount)'
        self._c.execute(sqlcommand, (simulationId, ))
        count = self._c.fetchall()
        assert len(count) == 1
        return count[0][0] == exceptedCount


    def _create_table_timeStepControlOneYear(self):
        """
        Create table timeStepControlOneYear
        """
        self._c.execute('''CREATE TABLE TimeStepControlOneYear (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, timestep INTEGER NOT NULL, dropCount INTEGER NOT NULL, overallTime REAL NOT NULL, timeOptionfile REAL NOT NULL, timeMetos3d REAL NOT NULL, timeMetos3dInit REAL NOT NULL, timeMetos3dSolver REAL NOT NULL, timeMetos3dFinal REAL NOT NULL, PRIMARY KEY (simulationId, year, timestep, dropCount))''')


    def insert_timeStepControlOneYear(self, simulationId, year, timestep, dropCount, overallTime, timeOptionfile, timeMetos3d, timeMetos3dInit, timeMetos3dSolver, timeMetos3dFinal, overwrite=False):
        """
        Insert values in the TimeStepControlOneYear table

        Insert time step control value for one year. If a time step control one
        year database entry for the simulationId, year, timestep and dropCount
        already exists, the existing entry is deleted and the new one is
        inserted (if the flag overwrite is True).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year of the spin up calculation
        timestep : {1, 2, 4, 8, 16, 32, 64}
            Time step used to estimate the local discretization error
        dropCount : int
            Number of iterations without accepting the step
        overallTime : float
            Overall time of the step size control step for one year
        timeOptionfile : float
            Time to create the optionfile for Metos3d
        timeMetos3d : float
            Time to run Metos3d (spin-up simulation)
        timeMetos3dInit : float
            Time to initialize Metos3d
        timeMetos3dSolver : float
            Time to run the Metos3d solver
        timeMetos3dFinal : float
            Time to finish Metos3d after the simulation
        overwrite : bool, default: False
            If True, overwrite an existing entry.

        Raises
        ------
        AssertionError
            If an existing entry exists and should not be overwritten.
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts.

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(dropCount) is int and 0 <= dropCount
        assert type(overallTime) is float and 0.0 < overallTime
        assert type(timeOptionfile) is float and 0.0 < timeOptionfile
        assert type(timeMetos3d) is float and 0.0 < timeMetos3d
        assert type(timeMetos3dInit) is float and 0.0 < timeMetos3dInit
        assert type(timeMetos3dSolver) is float and 0.0 < timeMetos3dSolver
        assert type(timeMetos3dFinal) is float and 0.0 < timeMetos3dFinal
        assert type(overwrite) is bool

        sqlcommand = 'SELECT simulationId, year, timestep, dropCount FROM TimeStepControlOneYear WHERE simulationId = ? AND year = ? AND timestep = ? AND dropCount = ?'
        self._c.execute(sqlcommand, (simulationId, year, timestep, dropCount))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM TimeStepControlOneYear WHERE simulationId = ? AND year = ? AND timestep = ? AND dropCount = ?'
                self._c.execute(sqlcommand, (simulationId, year, timestep, dropCount))
        else:
            assert len(dataset) == 0

        #Generate and insert timeStepControl value
        purchases = []
        purchases.append((simulationId, year, timestep, dropCount, overallTime, timeOptionfile, timeMetos3d, timeMetos3dInit, timeMetos3dSolver, timeMetos3dFinal))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO TimeStepControlOneYear VALUES (?,?,?,?,?,?,?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def checkTimeStepControlOneYear(self, simulationId, exceptedCount):
        """
        Check the number of entries in the table TimeStepControlOneYear

        Check the number of entries in the table TimeStepControlOneYear for
        the given simulationId. We use only the approximations of the first two
        time steps and omit the others approximations (in the case of dropped
        approximations).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        expectedCount : int, default: 1
            Expected number of database entries for the spin up

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(expectedCount) is int and expectedCount >= 0

        sqlcommand = 'SELECT COUNT(*) FROM TimeStepControlOneYear WHERE simulationId = ? AND dropCount = ?'
        self._c.execute(sqlcommand, (simulationId, 0))
        count = self._c.fetchall()
        assert len(count) == 1
        return count[0][0] == exceptedCount


    def _create_table_tracerDifferenceLhsNorm(self, norm='2'):
        """
        Create table TracerDifferenceLhs*Norm

        Parameters
        ----------
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        """
        assert norm in DB_Constants.NORM

        self._c.execute('''CREATE TABLE TracerDifferenceLhs{:s}Norm (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, timestepLhs INTEGER NOT NULL, tracer REAL NOT NULL, tracer_1dt REAL NOT NULL, N REAL NOT NULL, N_1dt REAL NOT NULL, DOP REAL, DOP_1dt REAL, P REAL, P_1dt REAL, Z REAL, Z_1dt REAL, D REAL, D_1dt REAL, PRIMARY KEY (simulationId, year, timestepLhs))'''.format(norm))


    def check_tracerDifferenceLhsNorm(self, simulationId, expectedCount, norm='2'):
        """
        Check number of tracer difference lhs norm entries for the simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        expectedCount : int
            Expected number of database entries for the spin up
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(expectedCount) is int and expectedCount >= 0
        assert norm in DB_Constants.NORM

        sqlcommand = 'SELECT COUNT(*) FROM TracerDifferenceLhs{}Norm WHERE simulationId = ?'.format(norm)
        self._c.execute(sqlcommand, (simulationId,))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***CheckTracerDifferenceLhsNorm: Expected: {} Get: {} (Norm: {})***'.format(expectedCount, count[0][0], norm))

        return count[0][0] == expectedCount


    def insert_difference_tracer_norm_tuple(self, simulationId, year, timestepLhs, tracerDifferenceNorm, tracerDifferenceNorm1dt, N, N1dt, DOP=None, DOP1dt=None, P=None, P1dt=None, Z=None, Z1dt=None, D=None, D1dt=None, norm='2', overwrite=False):
        """
        Insert the norm of a difference lhs between two tracers

        Insert the norm of a difference between two tracers. If a database
        entry of the norm between the tracers of the simulations with the
        simulationId, year and timestepLhs already exists, the existing
        entry is deleted and the new one is inserted (if the flag overwrite
        is True).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
            used tracer
        year : int
            Model year of the spin up calculation
        timestepLhs : int
            Time step used for the difference calculation
        tracerDifferenceNorm : float
            Norm including all tracers
        tracerDifferenceNorm1dt : float
            Norm including all tracers using time step 1dt
        N : float
            Norm of the N tracer
        N1dt : float
            Norm of the N tracer using time step 1dt
        DOP : None or float, default: None
            Norm of the DOP tracer. None, if the biogeochemical model does not
            contain the DOP tracer
        DOP1dt : None or float, default: None
            Norm of the DOP tracer using 1dt. None, if the biogeochemical model
            does not contain the DOP tracer
        P : None or float, default: None
            Norm of the P tracer. None, if the biogeochemical model does not
            contain the P tracer
        P1dt : None or float, default: None
            Norm of the P tracer using 1dt. None, if the biogeochemical model
            does not contain the P tracer
        Z : None or float, default: None
            Norm of the Z tracer. None, if the biogeochemical model does not
            contain the Z tracer
        Z1dt : None or float, default: None
            Norm of the Z tracer using 1dt. None, if the biogeochemical model
            does not contain the Z tracer
        D : None or float, default: None
            Norm of the D tracer. None, if the biogeochemical model does not
            contain the D tracer
        D1dt : None or float, default: None
            Norm of the D tracer using 1dt. None, if the biogeochemical model
            does not contain the D tracer
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        overwrite : bool, default: False
            If True, overwrite an existing entry.

        Raises
        ------
        AssertionError
            If an existing entry exists and should not be overwritten.
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts.

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert timestepLhs in Metos3d_Constants.METOS3D_TIMESTEPS 
        assert type(tracerDifferenceNorm) is float
        assert type(tracerDifferenceNorm1dt) is float
        assert type(N) is float
        assert type(N1dt) is float
        assert DOP is None or type(DOP) is float
        assert DOP1dt is None or type(DOP1dt) is float
        assert P is None or type(P) is float
        assert P1dt is None or type(P1dt) is float
        assert Z is None or type(Z) is float
        assert Z1dt is None or type(Z1dt) is float
        assert D is None or type(D) is float
        assert D1dt is None or type(D1dt) is float
        assert norm in DB_Constants.NORM
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationId, year, timestepLhs FROM TracerDifferenceLhs{}Norm WHERE simulationId = ? AND year = ? AND timestepLhs = ?'.format(norm)
        if overwrite:
            #Test, if dataset for this simulationIdA, year and timestepLhs combination exists
            self._c.execute(sqlcommand_select, (simulationId, year, timestepLhs))
            dataset = self._c.fetchall()
            #Remove database entry for this simulationId, year and timestepLhs
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM TracerDifferenceLhs{}Norm WHERE simulationId = ? AND year = ? AND timestepLhs = ?'.format(norm)
                self._c.execute(sqlcommand, (simulationId, year, timestepLhs))
        else:
            self._c.execute(sqlcommand_select, (simulationId, year, timestepLhs))
            dataset = self._c.fetchall()
            assert len(dataset) == 0

        #Generate insert for the tracer norm
        purchases = []
        purchases.append((simulationId, year, timestepLhs, tracerDifferenceNorm, tracerDifferenceNorm1dt, N, N1dt, DOP, DOP1dt, P, P1dt, Z, Z1dt, D, D1dt))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO TracerDifferenceLhs{}Norm VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'.format(norm), purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def read_rel_norm(self, metos3dModel, concentrationId, year=None, norm='2', parameterId=None, trajectory=''):
        """
        Returns the relative error

        Returns the relative error of all simulations using the given model
        and concentrationId. If parameterId is not None, this function returns
        only the relative difference for the given parameterId. If the year is
        not None, this function returns the relative error for the given year.

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        concentrationId : int
            Id of the concentration
        year : None or int, default: None
            Model year to return the relative error. If None, return the
            relative error for the last model year of the simulation.
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        parameterId : None or int, default: None
            Id of the parameter of the latin hypercube example. If None, this
            function returns the relative for all parameterIds.
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId and the relative error
        """
        pass


    def read_tracer_norm_values_for_simid(self, simulationId, norm='2', trajectory=''):
        """
        Returns norm values for the given simulationId

        Parameter
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory

        Returns
        -------
        numpy.ndarray
            2D array with the year and the norm value
        """
        assert type(simulationId) is int and simulationId >= 0
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'SELECT year, tracer FROM Tracer{}{}Norm WHERE simulationId = ? ORDER BY year;'.format(trajectory, norm)
        self._c.execute(sqlcommand,  (simulationId, ))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1
        return simdata


    def read_tracer_norm_value_for_simid_year(self, simulationId, year, norm='2', trajectory=''):
        """
        Return norm value for the given simulationId and year

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year of the calculated tracer concentration
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory

        Returns
        -------
        float
            Norm value

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'SELECT tracer FROM Tracer{}{}Norm WHERE simulationId = ? AND year = ?'.format(trajectory, norm)
        self._c.execute(sqlcommand,  (simulationId, year))
        simdata = self._c.fetchall()
        assert len(simdata) == 1
        return simdata[0][0]


    def read_tracer_difference_norm_values_for_simid(self, simulationId, simulationIdB, yearB=None, norm='2', trajectory=''):
        """
        Returns norm values of the difference for two simulations

        Returns the norm values for the difference of the tracer concentration
        between the spin up calculations with simulationId and simulationIdB
        as reference solution.

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        simulationIdB : int
            Id defining the parameter for spin up calculation for the reference
            solution
        yearB : int or None, default: None
            Model year of the calculated tracer concentration for the reference
            solution. If None, use the same year for both spin up calculations
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory

        Returns
        -------
        numpy.ndarray
            2D array with the year and the norm value
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(simulationIdB) is int and simulationIdB >= 0
        assert yearB is None or type(yearB) is int and 0 <= yearB
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        if yearB is None:
            sqlcommand = 'SELECT yearA, tracer FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ? AND yearA = yearB ORDER BY yearA;'.format(trajectory, norm)
            self._c.execute(sqlcommand,  (simulationId, simulationIdB))
        else:
            sqlcommand = 'SELECT yearA, tracer FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ? AND yearB = ? ORDER BY yearA;'.format(trajectory, norm)
            self._c.execute(sqlcommand,  (simulationId, simulationIdB, yearB))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1
        return simdata


    def read_spinupNorm_relNorm_for_model_year(self, model, startTimestep=1, stepYear=1, tolerance=1.0, rho=1.0, eta=2.0, cpus=128, normStepSizeControl='BoxweightedVol', checkConcentration=False, singleStep=False, singleStepYear=False, year=10000, norm='2', trajectory='', lhs=True):
        """
        Returns spin up norm and norm values

        Returns the spin up norm values and the relative error (the norm of
        the tracer concentration difference between the spin up calculation
        with step size control and the spin up calculation using the time
        step 1dt) for every parameter of the given model.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        startTimestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Initial time step for the step size control
        stepYear : int, default: 1
            Number of model years without adapting the time step
        tolerance : float, default: 1.0
            Tolerance for the local discretization error estimation
        rho : float, default: 1.0
            Factor for the tolerance used in the calculation of the
            optimial step size
        eta : float, default: 1.0
            Factor for scaling the step size for the next step
        cpus : int, default: 64
            Number of used cpus
        normStepSizeControl : {'2', 'BoxweightedVol', 'Boxweighted'},
                              default: 'BoxweightedVol'
            Used norm to estimate the local discretization error
        checkConcentration : bool, default: False
            If True, accept only concentrations with not negative
            concentrations
        singleStep : bool, default: False
            If True, use a single step (instead of model years) for the step
            size control
        singleStepYear : bool, default: False
            If True, the step size control used at the end of each model year
            the step size to calculate the concentration for the first time
            instant in the next model year. Otherwise, it is possible that the
            step size control does not compute the concentration at the first
            time instant.
        year : int, default: 10000
            Model year of the calculated tracer concentration for the spin up
            calculation. For the spin up norm the previous model year is used.
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        lhs : bool, default: True
            Use only the model parameter of the latin hypercube sample

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId, parameterId, the spin up norm and
            the relative error
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert startTimestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(stepYear) is int and 0 < stepYear
        assert type(tolerance) is float and 0.0 < tolerance
        assert type(rho) is float and 0.0 < rho
        assert type(eta) is float and 0.0 < eta
        assert type(cpus) is int and 0 < cpus
        assert normStepSizeControl in DB_Constants.NORM
        assert type(checkConcentration) is bool
        assert type(singleStep) is bool
        assert type(singleStepYear) is bool
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId, sim.parameterId, sp.toleranceTimestepSmall, normDiff.tracer/norm.tracer FROM Simulation AS sim, Simulation AS sim2, Spinup AS sp, StepControl AS sc, Tracer{:s}Norm AS norm, TracerDifference{:s}{:s}Norm AS normDiff WHERE sim.model = ? AND sim.startTimestep = ? AND sim.stepYear = ? AND sim.tolerance = ? AND sim.rho = ? AND sim.eta = ? AND sim.cpus = ? AND sim.norm = ? AND sim.checkConcentration = ? AND sim.singleStep = ? AND sim.singleStepYear = ?{:s} AND sim.simulationId = sp.simulationId AND sp.year = ? AND sim.simulationId = sc.simulationId AND sp.year = sc.year AND sp.timestepSmall = sc.timestepSmall AND sc.accept = ? AND sim2.simulationId = norm.simulationId AND norm.year = ? AND normDiff.simulationIdA = sim.simulationId AND normDiff.simulationIdB = sim2.simulationId AND normDiff.yearA = ? AND normDiff.yearB = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND sim2.startTimestep = ? AND sim2.stepYear = ? AND sim2.tolerance = ? AND sim2.rho = ? AND sim2.eta = ? ORDER BY sim.parameterId;'.format(norm, trajectory, norm, parameterStr)
        self._c.execute(sqlcommand, (model, startTimestep, stepYear, tolerance, rho, eta, cpus, normStepSizeControl, int(checkConcentration), int(singleStep), int(singleStepYear), year-1, int(True), year, year, 10000, 1, 10000, 1.0, 1.0, 2.0))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 4))

        i = 0
        for row in simrows:
            simdata[i, 0] = row[0]
            simdata[i, 1] = row[1]
            simdata[i, 2] = row[2]
            simdata[i, 3] = row[3]
            i = i+1

        return simdata


    def get_table_stepYear_value(self, metos3dModel, parameterId, concentrationId, startTimestep=1, stepYear=1, tolerance=1.0, rho=1.0, eta=2.0, cpus=128, norm='BoxweightedVol', checkConcentration=False, singleStep=False, singleStepYear=False, upperLayer=None, lowerLayer=None, year=10000, normError='Boxweighted', difference='Difference', endpoint=False, lhs=False):
        """
        Returns relative error of a simulation

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        concentrationId : int
            Id of the concentration
        startTimestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Initial time step for the step size control
        stepYear : int, default: 1
            Number of model years using the same time step
        tolerance : float, default: 1.0
            Tolerance for the local discretization error estimation
        rho : float, default: 1.0
            Factor for the tolerance used in the calculation of the
            optimial step size
        eta : float, default: 1.0
            Factor for scaling the step size for the next step
        cpus : int, default: 128
            Number of used cpus
        norm : {'2', 'BoxweightedVol', 'Boxweighted'}, default: 'BoxweightedVol'
            Used norm to estimate the local discretization error
        checkConcentration : bool, default: False
            If True, accept only concentrations with not negative
            concentrations
        singleStep : bool, default: False
            If True, use a single step (instead of model years) for the step
            size control
        singleStepYear : bool, default: False
            If True, the step size control used at the end of each model year
            the step size to calculate the concentration for the first time
            instant in the next model year. Otherwise, it is possible that the
            step size control does not compute the concentration at the first
            time instant.
        upperLayer : None or int, default: None
            If the value is an int, restrict the norm to the layer in ocean
            using the upperLayer index 'upperLayer'.
        lowerLayer : None or int, default: None
            If the value is an int, restrict the norm to the layer in ocean
            using the lowerLayer index 'lowerLayer'.
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and 0 <= concentrationId
        assert startTimestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(stepYear) is int and 0 < stepYear
        assert type(tolerance) is float and 0.0 < tolerance
        assert type(rho) is float and 0.0 < rho
        assert type(eta) is float and 0.0 < eta
        assert type(cpus) is int and 0 < cpus
        assert norm in DB_Constants.NORM
        assert type(checkConcentration) is bool
        assert type(singleStep) is bool
        assert type(singleStepYear) is bool
        assert upperLayer is None or type(upperLayer) is int and 0 <= upperLayer
        assert lowerLayer is None or type(lowerLayer) is int and 0 <= lowerLayer
        assert year is not None and year >= 0

        assert difference in ['', 'Difference']
        assert endpoint or not endpoint
        assert not (difference == '' and endpoint)
        assert lhs or not lhs
        assert (not lhs) or (lhs and difference)

        if (upperLayer is None and lowerLayer is None):
            upperLayer = 0
            lowerLayer = 14

        if lhs:
            lhs_str = 'Lhs'
        else:
            lhs_str = ''

        if difference:
            difference_str = 'Difference'
            if endpoint:
                endpoint_str = '_Endpoint' if (not lhs) else '_1dt'
            else:
                endpoint_str = ''
        else:
            difference_str = ''
            endpoint_str = ''

        sqlcommand = 'SELECT sim.simulationId as simid, diff_norm.tracer{}/norm.tracer as Error FROM Tracer{}{}{}Norm AS diff_norm, simulation AS sim, simulation AS sim2, tracer{}Norm AS norm WHERE sim.model = ? AND sim.parameterId = ? AND sim.concentrationId = ? AND sim.startTimestep = ? AND sim.stepYear = ? AND sim.tolerance = ? AND sim.rho = ? AND sim.eta = ? AND sim.cpus = ? AND sim.norm = ? AND sim.checkConcentration = ? AND sim.singleStep = ? AND sim.singleStepYear = ? AND sim.upperLayer = ? AND sim.lowerLayer = ? AND sim.simulationId = diff_norm.simulationId AND diff_norm.year = ? AND sim2.stepYear = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationID = sim2.concentrationID AND sim2.startTimestep = ? AND sim2.checkConcentration = ? AND sim2.singleStep = ? AND sim2.singleStepYear = ? AND sim2.upperLayer = ? AND sim2.lowerLayer = ? AND sim2.simulationId = norm.simulationId AND norm.year = ?'.format(endpoint_str, difference_str, lhs_str, normError, normError)
        self._c.execute(sqlcommand,  (metos3dModel, parameterId, concentrationId, startTimestep, stepYear, tolerance, rho, eta, cpus, norm, int(checkConcentration), int(singleStep), int(singleStepYear), upperLayer, lowerLayer, year, 10000, 1, 0, 0, 0, 0, 14, year))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows)))

        i = 0
        for row in simrows:
            simdata[i] = row[1]
            i = i+1

        return simdata

