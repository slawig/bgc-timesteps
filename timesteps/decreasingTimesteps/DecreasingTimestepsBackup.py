#!/usr/bin/env python
# -*- coding: utf8 -*

import bz2
import errno
import logging
import itertools
import os
import shutil
import tarfile

import metos3dutil.metos3d.constants as Metos3d_Constants
import decreasingTimesteps.constants as DecreasingTimesteps_Constants


class DecreasingTimestepsBackup():
    """
    Class for the backup using decreasing time steps
    """

    def __init__(self, parameterId=0, simulationList=None, movetar=False):
        """
        Initializes the backup using decreasing time steps

        Parameters
        ----------
        parameterId : int, default: 0
            Id of the parameter of the latin hypercube example
        simulationList : None or list [tuple]
            If None, use the default combination generated with the function
            _getSimulationList(). Otherwise, the list contains the tuples of
            model, timestep, tolerance, yearInterval and concentrationId
        movetar : bool
            If True, move the backup to/from the TAPE
        """
        assert type(parameterId) is int and parameterId in range(DecreasingTimesteps_Constants.PARAMETERID_MAX+1)
        assert simulationList is None or type(simulationList) is list
        assert type(movetar) is bool

        self._movetar = movetar
        self._parameterId = parameterId

        self._years = 10000
        self._trajectoryYear = 50
        self._decreasingTimestepsPath = os.path.join(DecreasingTimesteps_Constants.PATH, 'DecreasingTimesteps')

        self._simulationList = self._getSimulationList() if simulationList is None else simulationList


    def _getSimulationList(self):
        """
        Return the all parameter combinations for this parameterId

        Returns the combination of metos3dModel, timestep, tolerance,
        yearInterval and concentrationId used for different simulation runs

        Return
        ------
        list [tuple]
            List with tuples containing the model, timestep, tolerance and
            yearInterval. The list contains all combinations for the default
            parameter.
        """
        #Default values
        models = Metos3d_Constants.METOS3D_MODELS
        timestep = [64]
        tolerance = [0.001, 0.0001]
        yearInterval = [50, 100, 500]
        concentrationId = [None]

        simulationList = list(itertools.product(models, timestep, tolerance, yearInterval, concentrationId))

        if self._parameterId == 0:
            simulationList.extend(list(itertools.product(models, timestep, [0.01, 0.005, 0.0005, 0.00005, 0.00001], [yearInterval[0]], concentrationId)))

        return simulationList


    def backup(self):
        """
        Create backup include all files for the given parameterId

        The backup contains the files of the simulation using decreasing
        timesteps for all biogeochemical models.
        """
        act_path = os.getcwd()
        os.chdir(self._decreasingTimestepsPath)

        tarfilename = os.path.join(self._decreasingTimestepsPath, DecreasingTimesteps_Constants.PATTERN_BACKUP_FILENAME.format(self._parameterId, DecreasingTimesteps_Constants.COMPRESSION))
        assert not os.path.exists(tarfilename)
        tar = tarfile.open(tarfilename, 'w:{}'.format(DecreasingTimesteps_Constants.COMPRESSION), compresslevel=DecreasingTimesteps_Constants.COMPRESSLEVEL)

        for (metos3dModel, timestep, tolerance, yearInterval, concentrationId) in self._simulationList:
            path = os.path.join(metos3dModel, 'Parameter_{:0>3d}'.format(self._parameterId), '{:d}dt'.format(timestep), 'Years_{:0>5d}'.format(yearInterval), 'Tolerance_{:.1e}'.format(tolerance))

            #Add the logfile to the backup
            logfileName = DecreasingTimesteps_Constants.PATTERN_LOGFILE.format(metos3dModel, self._parameterId, timestep, concentrationId if concentrationId is not None else DecreasingTimesteps_Constants.CONCENTRATIONID_DICT[metos3dModel], tolerance, yearInterval)
            logfile = os.path.join(self._decreasingTimestepsPath, 'Logfile', logfileName)
            if os.path.exists(logfile) and os.path.isfile(logfile):
                #Copy logfile into simulation directory
                logfileBackup = os.path.join(path, logfileName)
                if not os.path.exists(os.path.join(self._decreasingTimestepsPath, logfileBackup)):
                    shutil.copy2(logfile, os.path.join(self._decreasingTimestepsPath, logfileBackup))

                #Backup of the logfile
                try:
                    tar.add(logfileBackup)
                except tarfile.TarError:
                    logging.warning('Can not add the log file {} to archiv'.format(logfileBackup))
            else:
                logging.warning('Log file {} does not exists'.format(logfile))

            #Add the job outputs to the backup
            for year in range(yearInterval, self._years+1, yearInterval):
                joboutput = os.path.join(path, DecreasingTimesteps_Constants.PATTERN_OUTPUT_FILENAME.format(year))
                if os.path.exists(os.path.join(self._decreasingTimestepsPath, joboutput)) and os.path.isfile(os.path.join(self._decreasingTimestepsPath, joboutput)):
                    try:
                        tar.add(joboutput)
                    except tarfile.TarError:
                        logging.warning('Can not add the joboutput file {} to archiv'.format(joboutput))
                else:
                    logging.warning('Joboutput file {} does not exists'.format(joboutput))

            #Add the tracer to the backup
            timestepIndex = -1
            for year in range(yearInterval, self._years+1, yearInterval):
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel]:
                    tracerName = os.path.join(path, 'Tracer', DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(year, tracer, Metos3d_Constants.METOS3D_TIMESTEPS[timestepIndex]))
                    while not os.path.exists(os.path.join(self._decreasingTimestepsPath, tracerName)) and tracer == Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel][0] and abs(timestepIndex) < len(Metos3d_Constants.METOS3D_TIMESTEPS):
                        timestepIndex -= 1
                        tracerName = os.path.join(path, 'Tracer', DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(year, tracer, Metos3d_Constants.METOS3D_TIMESTEPS[timestepIndex]))

                    for end in ['', '.info']:
                        tracerNameBackup = '{}{}'.format(tracerName, end)
                        if os.path.exists(os.path.join(self._decreasingTimestepsPath, tracerNameBackup)) and os.path.isfile(os.path.join(self._decreasingTimestepsPath, tracerNameBackup)):
                            try:
                                tar.add(tracerNameBackup)
                            except tarfile.TarError:
                                logging.warning('Can not add tracer file {} to archiv'.format(tracerNameBackup))
                        else:
                            logging.warning('Tracer file {} does not exist.'.format(tracerNameBackup))

            #Add the tracerOnestep to the backup
            for year in range(self._trajectoryYear, self._years+1, self._trajectoryYear):
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel]:
                    for end in ['', '.info']:
                        tracerName = os.path.join(path, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer), end))
                        if os.path.exists(os.path.join(self._decreasingTimestepsPath, tracerName)) and os.path.isfile(os.path.join(self._decreasingTimestepsPath, tracerName)):
                            try:
                                tar.add(tracerName)
                            except tarfile.TarError:
                                logging.warning('Can not add tracer file {} to archiv'.format(tracerName))
                        else:
                            logging.warning('Tracer file {} does not exist.'.format(tracerName))

        tar.close()

        #Move tarfile to TAPE_CACHE
        if self._movetar:
            shutil.move(tarfilename, os.path.join(DecreasingTimesteps_Constants.PATH_BACKUP, DecreasingTimesteps_Constants.PATTERN_BACKUP_FILENAME.format(self._parameterId, DecreasingTimesteps_Constants.COMPRESSION)))

        os.chdir(act_path)


    def restore(self, restoreLogfile=True, restoreJoboutput=True, restoreTracer=True, restoreTracerOnestep=True):
        """
        Restore the files of the simulation using decreasing timesteps.

        Parameter
        ---------
        restoreLogfile : bool, default: True
            If True, restore the log file
        restoreJoboutput : bool, default: True
            If True, restore the job outputs
        restoreTracer : bool, default: True
            If True, restore the tracer
        restoreTracerOnestep : bool, default: True
            If True, restore the tracerOnestep
        """
        assert type(restoreLogfile) is bool
        assert type(restoreJoboutput) is bool
        assert type(restoreTracer) is bool
        assert type(restoreTracerOnestep) is bool

        act_path = os.getcwd()
        os.chdir(self._decreasingTimestepsPath)

        tarfilename = os.path.join(self._decreasingTimestepsPath, DecreasingTimesteps_Constants.PATTERN_BACKUP_FILENAME.format(self._parameterId, DecreasingTimesteps_Constants.COMPRESSION))

        #Copy backup file
        if not os.path.exists(tarfilename) and self._movetar:
            shutil.copy2(os.path.join(DecreasingTimesteps_Constants.PATH_BACKUP, DecreasingTimesteps_Constants.PATTERN_BACKUP_FILENAME.format(self._parameterId, DecreasingTimesteps_Constants.COMPRESSION)), tarfilename)

        assert os.path.exists(tarfilename) and os.path.isfile(tarfilename)
        tar = tarfile.open(tarfilename, 'r:{}'.format(DecreasingTimesteps_Constants.COMPRESSION), compresslevel=DecreasingTimesteps_Constants.COMPRESSLEVEL)

        #Restore files
        for (metos3dModel, timestep, tolerance, yearInterval, concentrationId) in self._simulationList:
            path = os.path.join(metos3dModel, 'Parameter_{:0>3d}'.format(self._parameterId), '{:d}dt'.format(timestep), 'Years_{:0>5d}'.format(yearInterval), 'Tolerance_{:.1e}'.format(tolerance))

            os.makedirs(os.path.join(self._decreasingTimestepsPath, path), exist_ok=True)

            #Restore the log file
            if restoreLogfile:
                logfileName = DecreasingTimesteps_Constants.PATTERN_LOGFILE.format(metos3dModel, self._parameterId, timestep, concentrationId if concentrationId is not None else DecreasingTimesteps_Constants.CONCENTRATIONID_DICT[metos3dModel], tolerance, yearInterval)
                logfileBackup = os.path.join(path, logfileName)
                try:
                    tar.extract(logfileBackup, path=self._decreasingTimestepsPath)
                except (tarfile.TarError, KeyError):
                    logging.warning('There do not exist the log file {} in the archiv'.format(logfileBackup))

            #Restore the job outputs
            if restoreJoboutput:
                for year in range(yearInterval, self._years+1, yearInterval):
                    joboutput = os.path.join(path, DecreasingTimesteps_Constants.PATTERN_OUTPUT_FILENAME.format(year))
                    try:
                        tar.extract(joboutput, path=self._decreasingTimestepsPath)
                    except (tarfile.TarError, KeyError):
                        logging.warning('The job output {} does not exist in the archiv'.format(joboutput))

            #Restore the tracer
            if restoreTracer:
                os.makedirs(os.path.join(self._decreasingTimestepsPath, path, 'Tracer'), exist_ok=True)

                timestepIndex = -1
                for year in range(yearInterval, self._years+1, yearInterval):
                    for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel]:
                        tracerName = os.path.join(path, 'Tracer', DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(year, tracer, Metos3d_Constants.METOS3D_TIMESTEPS[timestepIndex]))

                        fileNotinTar = True
                        while fileNotinTar:
                            try:
                                tar.getmember(tracerName)
                                fileNotinTar = False
                            except KeyError:
                                timestepIndex -= 1
                                tracerName = os.path.join(path, 'Tracer', DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(year, tracer, Metos3d_Constants.METOS3D_TIMESTEPS[timestepIndex]))
                                
                        for end in ['', '.info']:
                            tracerNameBackup = '{}{}'.format(tracerName, end)
                            try:
                                tar.extract(tracerNameBackup, path=self._decreasingTimestepsPath)
                            except (tarfile.TarError, KeyError):
                                logging.warning('The tracer file {} is not in the archiv'.format(tracerNameBackup))

            #Restore the tracerOnestep
            if restoreTracerOnestep:
                os.makedirs(os.path.join(self._decreasingTimestepsPath, path, 'TracerOnestep'), exist_ok=True)

                for year in range(self._trajectoryYear, self._years+1, self._trajectoryYear):
                    for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel]:
                        for end in ['', '.info']:
                            tracerName = os.path.join(path, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer), end))
                            try:
                                tar.extract(tracerName, path=self._decreasingTimestepsPath)
                            except (tarfile.TarError, KeyError):
                                logging.warning('The tracer file {} is not in the archiv'.format(tracerName))

        tar.close()
        os.chdir(act_path)

        #Remove tarfile
        if self._movetar:
            os.remove(tarfilename)


    def remove(self):
        """
        Remove the files of the simulation using decreasing timesteps.
        """
        tarfilename = os.path.join(self._decreasingTimestepsPath, DecreasingTimesteps_Constants.PATTERN_BACKUP_FILENAME.format(self._parameterId, DecreasingTimesteps_Constants.COMPRESSION))

        #Copy backup file
        if not os.path.exists(tarfilename) and self._movetar:
            shutil.copy2(os.path.join(DecreasingTimesteps_Constants.PATH_BACKUP, DecreasingTimesteps_Constants.PATTERN_BACKUP_FILENAME.format(self._parameterId, DecreasingTimesteps_Constants.COMPRESSION)), tarfilename)

        assert os.path.exists(tarfilename) and os.path.isfile(tarfilename)
        tar = tarfile.open(tarfilename, 'r:{}'.format(DecreasingTimesteps_Constants.COMPRESSION), compresslevel=DecreasingTimesteps_Constants.COMPRESSLEVEL)

        #Remove files
        for (metos3dModel, timestep, tolerance, yearInterval, concentrationId) in self._simulationList:
            path = os.path.join(metos3dModel, 'Parameter_{:0>3d}'.format(self._parameterId), '{:d}dt'.format(timestep), 'Years_{:0>5d}'.format(yearInterval), 'Tolerance_{:.1e}'.format(tolerance))

            #Remove log file
            logfileName = DecreasingTimesteps_Constants.PATTERN_LOGFILE.format(metos3dModel, self._parameterId, timestep, concentrationId if concentrationId is not None else DecreasingTimesteps_Constants.CONCENTRATIONID_DICT[metos3dModel], tolerance, yearInterval)
            logfile = os.path.join(self._decreasingTimestepsPath, 'Logfile', logfileName)
            logfileBackup = os.path.join(path, logfileName)
            try:
                info = tar.getmember(logfileBackup)
            except KeyError:
                logging.info('There does not exist the log file {} in the archiv'.format(logfileBackup))
            else:
                if info.isfile() and info.size > 0 and os.path.exists(os.path.join(self._decreasingTimestepsPath, logfileBackup)) and os.path.isfile(os.path.join(self._decreasingTimestepsPath, logfileBackup)):
                    os.remove(os.path.join(self._decreasingTimestepsPath, logfileBackup))
                    if os.path.exists(logfile) and os.path.isfile(logfile):
                        os.remove(logfile)

            #Remove job output
            joboutput = os.path.join(self._decreasingTimestepsPath, 'Joboutput', DecreasingTimesteps_Constants.PATTERN_JOBOUTPUT.format(metos3dModel, self._parameterId, timestep, concentrationId if concentrationId is not None else DecreasingTimesteps_Constants.CONCENTRATIONID_DICT[metos3dModel], tolerance, yearInterval))
            if os.path.exists(joboutput) and os.path.isfile(joboutput):
                os.remove(joboutput)

            #Remove metos3d job outputs
            for year in range(yearInterval, self._years+1, yearInterval):
                joboutput = os.path.join(path, DecreasingTimesteps_Constants.PATTERN_OUTPUT_FILENAME.format(year))
                try:
                    info = tar.getmember(joboutput)
                except KeyError:
                    logging.info('There does not exist the job output file {} in the archiv'.format(joboutput))
                else:
                    if info.isfile() and info.size > 0 and os.path.exists(os.path.join(self._decreasingTimestepsPath, joboutput)) and os.path.isfile(os.path.join(self._decreasingTimestepsPath, joboutput)):
                        os.remove(os.path.join(self._decreasingTimestepsPath, joboutput))

            #Remove tracer
            if os.path.exists(os.path.join(self._decreasingTimestepsPath, path, 'Tracer')) and os.path.isdir(os.path.join(self._decreasingTimestepsPath, path, 'Tracer')):
                timestepIndex = -1
                for year in range(yearInterval, self._years+1, yearInterval):
                    for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel]:
                        tracerName = os.path.join(path, 'Tracer', DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(year, tracer, Metos3d_Constants.METOS3D_TIMESTEPS[timestepIndex]))
                        while not os.path.exists(os.path.join(self._decreasingTimestepsPath, tracerName)) and abs(timestepIndex) <= len(Metos3d_Constants.METOS3D_TIMESTEPS):
                            timestepIndex -= 1
                            tracerName = os.path.join(path, 'Tracer', DecreasingTimesteps_Constants.PATTERN_TRACER_OUTPUT_YEAR_TIMESTEP.format(year, tracer, Metos3d_Constants.METOS3D_TIMESTEPS[timestepIndex]))

                        for end in ['', '.info']:
                            tracerNameBackup = '{}{}'.format(tracerName, end)
                            try:
                                info = tar.getmember(tracerNameBackup)
                            except KeyError:
                                logging.info('There does not exist the tracer file {} in the archiv'.format(tracerNameBackup))
                            else:
                                if info.isfile() and info.size > 0 and os.path.exists(os.path.join(self._decreasingTimestepsPath, tracerNameBackup)) and os.path.isfile(os.path.join(self._decreasingTimestepsPath, tracerNameBackup)):
                                    os.remove(os.path.join(self._decreasingTimestepsPath, tracerNameBackup))

                #Remove the tracer directory
                try:
                    os.rmdir(os.path.join(self._decreasingTimestepsPath, path, 'Tracer'))
                except OSError as ex:
                    if ex.errno == errno.ENOTEMPTY:
                        logging.info('Tracer directory {} is not empty'.format(os.path.join(self._decreasingTimestepsPath, path, 'Tracer')))

            #Remove tracerOnestep
            if os.path.exists(os.path.join(self._decreasingTimestepsPath, path, 'TracerOnestep')) and os.path.isdir(os.path.join(self._decreasingTimestepsPath, path, 'TracerOnestep')):
                for year in range(self._trajectoryYear, self._years+1, self._trajectoryYear):
                    for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel]:
                        for end in ['', '.info']:
                            tracerName = os.path.join(path, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer), end))
                            try:
                                info = tar.getmember(tracerName)
                            except KeyError:
                                logging.info('There does not exist the tracer file {} in the archiv'.format(tracerName))
                            else:
                                if info.isfile() and info.size > 0 and os.path.exists(os.path.join(self._decreasingTimestepsPath, tracerName)) and os.path.isfile(os.path.join(self._decreasingTimestepsPath, tracerName)):
                                    os.remove(os.path.join(self._decreasingTimestepsPath, tracerName))

                #Remove the TracerOnestep directory
                try:
                    os.rmdir(os.path.join(self._decreasingTimestepsPath, path, 'TracerOnestep'))
                except OSError as ex:
                    if ex.errno == errno.ENOTEMPTY:
                        logging.info('Tracer directory {} is not empty'.format(os.path.join(self._decreasingTimestepsPath, path, 'TracerOnestep')))

            #Remove directories
            try:
                directoryPath = path
                while (directoryPath != '' and os.path.exists(os.path.join(self._decreasingTimestepsPath, directoryPath)) and os.path.isdir(os.path.join(self._decreasingTimestepsPath, directoryPath))):
                    if not os.listdir(os.path.join(self._decreasingTimestepsPath, directoryPath)):
                        os.rmdir(os.path.join(self._decreasingTimestepsPath, directoryPath))
                        directoryPath = os.path.dirname(directoryPath)
                    else:
                        break
            except OSError as ex:
                if ex.errno == errno.ENOTEMPTY:
                    logging.info('Directory {} is not empty'.format(directoryPath))

        tar.close()

        #Remove tarfile
        if self._movetar:
            os.remove(tarfilename)

