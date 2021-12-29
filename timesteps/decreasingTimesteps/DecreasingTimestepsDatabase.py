#!/usr/bin/env python
# -*- coding: utf8 -*

import logging
import numpy as np
import os
import sqlite3
import time

import metos3dutil.database.constants as DB_Constants
import metos3dutil.latinHypercubeSample.constants as LHS_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.database.DatabaseMetos3d import DatabaseMetos3d
import decreasingTimesteps.constants as DecreasingTimesteps_Constants


class DecreasingTimestepsDatabase(DatabaseMetos3d):
    """
    Access functions for the database
    """

    def __init__(self, dbpath=DecreasingTimesteps_Constants.DB_PATH, completeTable=True, createDb=False):
        """
        Initialization of the database connection

        Parameters
        ----------
        dbpath : str, default: decreasingTimesteps.constants.DB_PATH
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
        self._create_table_convergence()
        self._create_table_timesteps()

        #Database tables for the norm
        for norm in DB_Constants.NORM:
            self._create_table_tracerNorm(norm=norm)
            self._create_table_tracerNorm(norm=norm, trajectory='Trajectory')
            self._create_table_tracerDifferenceNorm(norm=norm)
            self._create_table_tracerDifferenceNorm(norm=norm, trajectory='Trajectory')

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
        self._c.execute('''CREATE TABLE Simulation (simulationId INTEGER NOT NULL, model TEXT NOT NULL, parameterId INTEGER NOT NULL REFERENCES Parameter(parameterId), concentrationId INTEGER NOT NULL REFERENCES InitialConcentration(concentrationId), timestep INTEGER NOT NULL, yearInterval INTEGER NOT NULL, tolerance REAL NOT NULL, UNIQUE (model, parameterId, concentrationId, timestep, yearInterval, tolerance), PRIMARY KEY (simulationId))''')


    def _init_table_simulation(self):
        """
        Initial insert of simulation data sets
        """
        concentrationDic = {'N': 0, 'N-DOP': 1, 'NP-DOP': 2, 'NPZ-DOP': 3, 'NPZD-DOP': 4, 'MITgcm-PO4-DOP': 1}

        purchases = []
        simulationId = 0

        #Simulation of the reference solutions
        timestep = 1
        yearInterval = 10000
        tolerance = 0.01
        for parameterId in range(DecreasingTimesteps_Constants.PARAMETERID_MAX+1):
            for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
                purchases.append((simulationId, metos3dModel, parameterId, concentrationDic[metos3dModel], timestep, yearInterval, tolerance))
                simulationId += 1

        #Simulation with different tolerance
        parameterId = 0
        timestep = Metos3d_Constants.METOS3D_TIMESTEPS[-1]
        yearInterval = 50
        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            for tolerance in [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]:
                purchases.append((simulationId, metos3dModel, parameterId, concentrationDic[metos3dModel], timestep, yearInterval, tolerance))
                simulationId += 1

        #Simulation for different parameter
        for timestep in [Metos3d_Constants.METOS3D_TIMESTEPS[-1]]:
            for parameterId in range(DecreasingTimesteps_Constants.PARAMETERID_MAX+1):
                for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
                    for yearInterval in [50, 100, 500]:
                        for tolerance in [0.001, 0.0001]:
                            if (timestep == Metos3d_Constants.METOS3D_TIMESTEPS[-1] and parameterId == 0 and yearInterval == 50):
                                #Simulation for this combination already included (see above: Simulation with different tolerance)
                                pass
                            else:
                                purchases.append((simulationId, metos3dModel, parameterId, concentrationDic[metos3dModel], timestep, yearInterval, tolerance))
                                simulationId += 1

        self._c.executemany('INSERT INTO Simulation VALUES (?,?,?,?,?,?,?)', purchases)
        self._conn.commit()


    def exists_simulaiton(self, metos3dModel, parameterId, concentrationId, timestep=1, yearInterval=50, tolerance=0.01):
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
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation
        yearInterval : int, default: 50
            Number of model years of each spin up
        tolerance : float, default: 0.01
            Boarder for the error to decrease the time stepp

        Returns
        -------
        bool
            True if an entry exists for the given values
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and 0 <= concentrationId
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(yearInterval) is int and 0 <= yearInterval
        assert type(tolerance) is float and 0 < tolerance

        sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND parameterId = ? AND concentrationId = ? AND timestep = ? AND yearInterval = ? AND tolerance = ?'
        self._c.execute(sqlcommand, (metos3dModel, parameterId, concentrationId, timestep, yearInterval, tolerance))
        simulationId = self._c.fetchall()
        return len(simulationId) > 0


    def get_simulationId(self, metos3dModel, parameterId, concentrationId, timestep=1, yearInterval=50, tolerance=0.01):
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
            concentrationId, yearInterval and tolerance

        Raises
        ------
        AssertionError
            If no entry for the model, parameterId, concentrationId,
            timestep, yearInterval and tolerance exists in the database
            table Simulation
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and 0 <= concentrationId
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(yearInterval) is int and 0 <= yearInterval
        assert type(tolerance) is float and 0 < tolerance

        sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND parameterId = ? AND concentrationId = ? AND timestep = ? AND yearInterval = ? AND tolerance = ?'
        self._c.execute(sqlcommand, (metos3dModel, parameterId, concentrationId, timestep, yearInterval, tolerance))
        simulationId = self._c.fetchall()
        assert len(simulationId) == 1
        return simulationId[0][0]


    def insert_simulation(self, metos3dModel, parameterId, concentrationId, timestep=1, yearInterval=50, tolerance=0.01):
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
            concentrationId, yearInterval and tolerance

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
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(yearInterval) is int and 0 <= yearInterval
        assert type(tolerance) is float and 0 < tolerance

        if self.exists_simulaiton(metos3dModel, parameterId, concentrationId, timestep=timestep, yearInterval=yearInterval, tolerance=tolerance):
            #Simulation already exists in the database
            simulationId = self.get_simulationId(metos3dModel, parameterId, concentrationId, timestep=timestep, yearInterval=yearInterval, tolerance=tolerance)
        else:
            #Insert simulation into the database
            sqlcommand = 'SELECT MAX(simulationId) FROM Simulation'
            self._c.execute(sqlcommand)
            dataset = self._c.fetchall()
            assert len(dataset) == 1
            simulationId = dataset[0][0] + 1

            purchases = [(simulationId, metos3dModel, parameterId, concentrationId, timestep, yearInterval, tolerance)]
            inserted = False
            insertCount = 0
            while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
                try:
                    self._c.executemany('INSERT INTO Simulation VALUES (?,?,?,?,?,?,?)', purchases)
                    self._conn.commit()
                    inserted = True
                except sqlite3.OperationalError:
                    insertCount += 1
                    #Wait for the next insert
                    time.sleep(DB_Constants.TIME_SLEEP)

        return simulationId


    def _create_table_convergence(self):
        """
        Create table Convergence
        """
        self._c.execute('''CREATE TABLE Convergence (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), convergence INTEGER NOT NULL, PRIMARY KEY (simulationId))''')


    def check_convergence(self, simulationId):
        """
        Returns if an entries for the convergence exists

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        bool
           True if an database entry exists for the simulationId
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT convergence FROM Convergence WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        convergence = self._c.fetchall()
        return len(convergence) > 0


    def get_convergence(self, simulationId):
        """
        Returns the convergence behaviour

        Returns the convergence behaviour for the spin up calculation

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        
        Returns
        -------
        bool
            True in the case of a convergent spin up calculation otherwise
            False

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT convergence FROM Convergence WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        convergence = self._c.fetchall()
        assert len(convergence) == 1
        return bool(convergence[0][0])


    def insert_convergence(self, simulationId, convergence, overwrite=False):
        """
        Insert the convergence behaviour

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        convergence : bool
            True for a convergent spin up calculation otherwise False
        overwrite : bool, default: False
            If True, remove existing value for the simulationId, year and
            measurementId before insert

        Raises
        ------
        AssertionError
            If an entry exists for the simulationId in the database table
            Convergence using overwriting=False
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(convergence) is bool
        assert type(overwrite) is bool

        sqlcommand = 'SELECT simulationId FROM Convergence WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Convergence WHERE simulationId = ?'
                self._c.execute(sqlcommand, (simulationId, ))
        else:
            assert len(dataset) == 0

        #Insert for the tracer norm
        purchases = []
        purchases.append((simulationId, int(convergence)))
        self._c.executemany('INSERT INTO Convergence VALUES (?,?)', purchases)
        self._conn.commit()


    def _create_table_timesteps(self):
        """
        Create table Timesteps
        """
        self._c.execute('''CREATE TABLE Timesteps (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, timestep INTEGER NOT NULL, reduction REAL, accepted INTEGER NOT NULL, PRIMARY KEY (simulationId, year, timestep))''')


    def check_timesteps(self, simulationId, expectedCount):
        """
        Check the number of used time steps entries for the given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        expectedCount : int
            Expected number of database entries for the used time steps

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(expectedCount) is int and expectedCount >= 0

        sqlcommand = 'SELECT COUNT(*) FROM Timesteps WHERE simulationId = ? AND accepted = ?'
        self._c.execute(sqlcommand, (simulationId, int(True)))
        count = self._c.fetchall()
        assert len(count) == 1
        return count[0][0] == expectedCount


    def insert_timesteps(self, simulationId, year, timestep, reduction, accepted, overwrite=False):
        """
        Insert the convergence behaviour

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            First model year of the spin up for which we used the given time
            step
        timestep : {1, 2, 4, 8, 16, 32, 64}
            Used time step for the spin up started for the given year
        reduction : float or None
            Reduction between the tracer concentration before the spin up and
            at the end of the spin up (relative error). Use the value None, if
            the spin up simulations ends with NaN as concentration value in at
            least one box.
        accepted : bool
            If True accepted the spin up for the last time interval. Otherwise
            calculated the spin up again with a reduced time step
        overwrite : bool, default: False
            If True, remove existing value for the simulationId, year and
            measurementId before insert

        Raises
        ------
        AssertionError
            If an entry exists for the simulationId in the database table
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert reduction is None or type(reduction) is float and reduction >= 0
        assert type(accepted) is bool
        assert type(overwrite) is bool

        sqlcommand = 'SELECT simulationId, year, timestep FROM Timesteps WHERE simulationId = ? AND year = ? and timestep = ?'
        self._c.execute(sqlcommand, (simulationId, year, timestep))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Timesteps WHERE simulationId = ? AND year = ? AND timestep = ?'
                self._c.execute(sqlcommand, (simulationId, year, timestep))
        else:
            assert len(dataset) == 0

        #Insert for the tracer norm
        purchases = []
        purchases.append((simulationId, year, timestep, reduction, int(accepted)))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO Timesteps VALUES (?,?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def read_year_timesteps(self, simulationId):
        """
        Returns the year and the timestep for the given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        numpy.ndarray
            2D array with the year and the timestep
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT year, timestep FROM Timesteps WHERE simulationId = ? ORDER BY year;'
        self._c.execute(sqlcommand, (simulationId, ))
        dataset = self._c.fetchall()
        timestepData = np.zeros(shape=(len(dataset), 2))

        i = 0
        for row in dataset:
            timestepData[i,:] = np.array([row[0], row[1]])
            i += 1

        return timestepData


    def read_tracer_norm_values_for_simid(self, simulationId, norm='2', trajectory=''):
        """
        Returns norm values for the given simulationId

        Parameters
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


    def read_spinupNorm_relNorm_for_model_year(self, model, timestep=64, yearInterval=50, tolerance=0.001, year=10000, norm='2', trajectory='', lhs=True):
        """
        Returns spin up norm and norm values

        Returns the spin up norm values and the relative error (the norm of
        the tracer concentration difference between the spin up calculation
        with decreasing time steps and the spin up calculation using the time
        step 1dt) for every parameter of the given model.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 64
            Initial time step of the decreasing time step simulation
        yearInterval : int, default: 50
            Interval of model years using the same time step
        tolerance : float, default: 0.001
            Decrease the time step if the tolerance is undercut.
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
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(yearInterval) is int and 0 < yearInterval
        assert type(tolerance) is float and 0 < tolerance
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId, sim.parameterId, sp.tolerance, normDiff.tracer/norm.tracer FROM Simulation AS sim, Simulation AS sim2, Spinup AS sp, Tracer{:s}Norm AS norm, TracerDifference{:s}{:s}Norm AS normDiff WHERE sim.model = ? AND sim.timestep = ? AND sim.yearInterval = ? AND sim.tolerance = ?{:s} AND sim.simulationId = sp.simulationId AND sp.year = ? AND sim2.simulationId = norm.simulationId AND norm.year = ? AND normDiff.simulationIdA = sim.simulationId AND normDiff.yearA = ? AND normDiff.yearB = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND sim2.timestep = ? AND sim2.yearInterval = ? AND sim2.tolerance = ? AND normDiff.simulationIdB = sim2.simulationId ORDER BY sim.parameterId;'.format(norm, trajectory, norm, parameterStr)
        self._c.execute(sqlcommand, (model, timestep, yearInterval, tolerance, year-1, year, year, year, 1, 10000, 0.01))
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
        #TODO implement function
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
        #TODO implement function
        pass


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
        #TODO implement function
        pass

