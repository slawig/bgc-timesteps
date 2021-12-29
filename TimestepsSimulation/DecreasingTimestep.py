#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os
import sqlite3

import metos3dutil.metos3d.constants as Metos3d_Constants
import neshCluster.constants as NeshCluster_Constants
import decreasingTimesteps.constants as DecreasingTimesteps_Constants


from system.system import SYSTEM
if SYSTEM == 'PC':
    from standaloneComputer.JobAdministration import JobAdministration
else:
    from neshCluster.JobAdministration import JobAdministration


def main(metos3dModel, parameterIdList=range(DecreasingTimesteps_Constants.PARAMETERID_MAX+1), timestepList=Metos3d_Constants.METOS3D_TIMESTEPS, concentrationId=None, toleranceList=[0.01], yearIntervalList=[50], spinupTolerance=None, partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES, memory=None, time=None):
    """
    Create jobs for the spin ups using different time steps

    Parameters
    ----------
    metos3dModel : str
        Name of the biogeochemical model
    parameterIdList : list [int] or range,
        default: range(Timesteps_Constants.PARAMETERID_MAX+1)
        List of parameterIds of the latin hypercube example
    timestepList : list [int], default: [1, 2, 4, 8, 16, 32, 64]
        List of time steps of the spin up simulation
    concentrationId : int or None, default: None
        Id of the initial tracer concentration
    toleranceList : list [float], default: [0.01]
        List of tolerances to decrease the time step if the relative error of
        the last time interval is less than the tolerance
    yearIntervalList : list [int], default: [50]
        List of year intervals to test the reduction of the spin up simulation
        (relative error) after the given number of years
    spinupTolerance : float or None, default: None
        If not None, check if the spin up tolerance of the simulation
        using the standard constant concentration is less than the given
        spinupTolerance value
    partition : str, default: NeshCluster_Constants.DEFAULT_PARTITION
        Partition of the NEC HPC-Linux-Cluster of the CAU Kiel
    qos : str, default: NeshCluster_Constants.DEFAULT_QOS
        Quality of service of the NEC HPC-Linux-Cluster of the CAU Kiel
    nodes : int, default: NeshCluster_Constants.DEFAULT_NODES
        Number of nodes on the NEC HPC-Linux-Cluster of the CAU Kiel
    memory : int or None, default: None
        Reserved memory on the NEC HPC-Linux-Cluster of the CAU Kiel
    time : int or None, default: None
        Walltime in hours on the NEC HPC-Linux-Cluster of the CAU Kiel
    """
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(parameterIdList) in [list, range]
    assert type(timestepList) in [list, range]
    assert concentrationId is None or type(concentrationId) is int and 0 <= concentrationId
    assert type(toleranceList) in [list, range]
    assert type(yearIntervalList) in [list, range]
    assert spinupTolerance is None or type(spinupTolerance) is float and 0 < spinupTolerance
    assert partition in NeshCluster_Constants.PARTITION
    assert qos in NeshCluster_Constants.QOS
    assert type(nodes) is int and 0 < nodes
    assert memory is None or type(memory) is int and 0 < memory
    assert time is None or type(time) is int and 0 < time

    decreasingTimestepsSimulation = DecreasingTimestepsSimulation(metos3dModel, parameterIdList=parameterIdList, timestepList=timestepList, concentrationId=concentrationId, toleranceList=toleranceList, yearIntervalList=yearIntervalList, partition=partition, qos=qos, nodes=nodes, memory=memory, time=time)
    decreasingTimestepsSimulation.generateJobList(spinupTolerance=spinupTolerance)
    decreasingTimestepsSimulation.runJobs()



class DecreasingTimestepsSimulation(JobAdministration):
    """
    Administration of the jobs organizing spin ups using different time steps
    """

    def __init__(self, metos3dModel, parameterIdList=range(DecreasingTimesteps_Constants.PARAMETERID_MAX+1), timestepList=Metos3d_Constants.METOS3D_TIMESTEPS, concentrationId=None, toleranceList=[0.01], yearIntervalList=[50], partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES, memory=None, time=None):
        """
        Initializes the jobs of the spin ups using different time steps

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterIdList : list [int] or range,
            default: range(Timesteps_Constants.PARAMETERID_MAX+1)
            List of parameterIds of the latin hypercube example
        timestepList : list, default: [1, 2, 4, 8, 16, 32, 64]
            List of time steps of the spin up simulation
        concentrationId : int or None, default: None
            Id of the initial tracer concentration
        toleranceList : list [float], default: [0.01]
            List of tolerances to decrease the time step if the relative error
            of the last time interval is less than the tolerance
        yearIntervalList : list [int], default: [50]
            List of year intervals to test the reduction of the spin up
            simulation (relative error) after the given number of years
        partition : str, default: NeshCluster_Constants.DEFAULT_PARTITION
            Partition of the NEC HPC-Linux-Cluster of the CAU Kiel
        qos : str, default: NeshCluster_Constants.DEFAULT_QOS
            Quality of service of the NEC HPC-Linux-Cluster of the CAU Kiel
        nodes : int, default: NeshCluster_Constants.DEFAULT_NODES
            Number of nodes on the NEC HPC-Linux-Cluster of the CAU Kiel
        memory : int or None, default: None
            Reserved memory on the NEC HPC-Linux-Cluster of the CAU Kiel
        time : int or None, default: None
            Walltime in hours on the NEC HPC-Linux-Cluster of the CAU Kiel
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterIdList) in [list, range]
        assert type(timestepList) in [list, range]
        assert concentrationId is None or type(concentrationId) is int and 0 <= concentrationId
        assert type(toleranceList) in [list, range]
        assert type(yearIntervalList) in [list, range]
        assert partition in NeshCluster_Constants.PARTITION
        assert qos in NeshCluster_Constants.QOS
        assert type(nodes) is int and 0 < nodes
        assert memory is None or type(memory) is int and 0 < memory
        assert time is None or type(time) is int and 0 < time

        JobAdministration.__init__(self)

        self._metos3dModel = metos3dModel
        self._parameterIdList = parameterIdList
        self._timestepList = timestepList
        self._concentrationId = concentrationId
        self._toleranceList = toleranceList
        self._yearIntervalList = yearIntervalList

        self._partition = partition
        self._qos = qos
        self._nodes = nodes
        self._memory = memory
        self._time = time


    def generateJobList(self, spinupTolerance=None):
        """
        Generates a list of jobs of spin ups using different time steps

        Parameters
        ----------
        spinupTolerance : float or None, default: None
            If not None, check if the spin up tolerance of the simulation
            using the standard constant concentration is less than the given
            spinupTolerance value

        Notes
        -----
        Adds the jobs into the internal job list
        """
        assert spinupTolerance is None or type(spinupTolerance) is float and 0 < spinupTolerance

        for yearInterval in self._yearIntervalList:
            for tolerance in self._toleranceList:
                for parameterId in self._parameterIdList:
                    for timestep in self._timestepList:
                        if not self._checkJob(parameterId, timestep, tolerance, yearInterval):
                            program = 'DecreasingTimestep_Jobcontrol.py -metos3dModel {:s} -parameterId {:d} -timestep {:d} -tolerance {:f} -year {:d} -nodes {:d}'.format(self._metos3dModel, parameterId, timestep, tolerance, yearInterval, self._nodes)

                            #Optional parameter
                            if self._concentrationId is not None:
                                program += ' -concentrationId {:d}'.format(self._concentrationId)

                            jobDict = {}
                            jobDict['jobFilename'] = os.path.join(DecreasingTimesteps_Constants.PATH, 'DecreasingTimesteps', 'Jobfile', DecreasingTimesteps_Constants.PATTERN_JOBFILE.format(self._metos3dModel, parameterId, timestep, self._concentrationId if self._concentrationId is not None else DecreasingTimesteps_Constants.CONCENTRATIONID_DICT[self._metos3dModel], tolerance, yearInterval))
                            jobDict['path'] = os.path.join(DecreasingTimesteps_Constants.PATH, 'DecreasingTimesteps', 'Jobfile')
                            jobDict['jobname'] = 'DecreasingTimesteps_{}_{:d}_{:d}'.format(self._metos3dModel, parameterId, timestep)
                            jobDict['joboutput'] = os.path.join(DecreasingTimesteps_Constants.PATH, 'DecreasingTimesteps', 'Joboutput', DecreasingTimesteps_Constants.PATTERN_JOBOUTPUT.format(self._metos3dModel, parameterId, timestep, self._concentrationId if self._concentrationId is not None else DecreasingTimesteps_Constants.CONCENTRATIONID_DICT[self._metos3dModel], tolerance, yearInterval))
                            jobDict['programm'] = os.path.join(DecreasingTimesteps_Constants.PROGRAM_PATH, program)
                            jobDict['partition'] = self._partition
                            jobDict['qos'] = self._qos
                            jobDict['nodes'] = self._nodes
                            jobDict['pythonpath'] = DecreasingTimesteps_Constants.DEFAULT_PYTHONPATH
                            jobDict['loadingModulesScript'] = NeshCluster_Constants.DEFAULT_LOADING_MODULES_SCRIPT

                            if self._memory is not None:
                                jobDict['memory'] = self._memory
                            if self._time is not None:
                                jobDict['time'] = self._time

                            self.addJob(jobDict)


    def _checkJob(self, parameterId, timestep, tolerance, yearInterval):
        """
        Check if the job run already exists

        Parameters
        ----------
        parameterId : int
            Id of the parameter of the latin hypercube example
        timestep : {1, 2, 4, 8, 16, 32, 64}
            Time step of the spin up simulation
        tolerance: float
            Decrease the time step if the relative error of the last time
            interval is less than the tolerance
        yearInterval : int
            Test the reduction of the spin up simulation (relative error) after
            the given number of years 

        Returns
        -------
        bool
           True if the joboutput already exists
        """
        assert type(parameterId) is int and parameterId in range(DecreasingTimesteps_Constants.PARAMETERID_MAX+1)
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(tolerance) is float and 0.0 < tolerance
        assert type(yearInterval) is int and 0 < yearInterval

        joboutput = os.path.join(DecreasingTimesteps_Constants.PATH, 'DecreasingTimesteps', 'Joboutput', DecreasingTimesteps_Constants.PATTERN_JOBOUTPUT.format(self._metos3dModel, parameterId, timestep, self._concentrationId if self._concentrationId is not None else DecreasingTimesteps_Constants.CONCENTRATIONID_DICT[self._metos3dModel], tolerance, yearInterval))
        return os.path.exists(joboutput) and os.path.isfile(joboutput)


if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-metos3dModel', type=str, help='Name of the biogeochemical model')
    parser.add_argument('-parameterIds', nargs='*', type=int, default=[0], help='List of parameterIds')
    parser.add_argument('-parameterIdRange', nargs=2, type=int, default=[0], help='Create list parameterIds using range (-parameterIdRange a b: range(a, b)')
    parser.add_argument('-timesteps', nargs='*', type=int, default=Metos3d_Constants.METOS3D_TIMESTEPS, help='List of time steps for the spin up simulation')
    parser.add_argument('-concentrationId', nargs='?', const=None, default=None, help='Id of the initial concentration')
    parser.add_argument('-tolerance', nargs='*', type=float, default=[0.01], help='Boarder of the tolerance')
    parser.add_argument('-yearInterval', nargs='*', type=int, default=[50], help='Number of model years used for the spin up')
    parser.add_argument('-spinupTolerance', nargs='?', const=None, default=None, help='Simulation with another initial concentration if the simulation using standard concentration does not reach the spin up tolerance')
    parser.add_argument('-partition', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_PARTITION, default=NeshCluster_Constants.DEFAULT_PARTITION, help='Partition of slum on the Nesh-Cluster (Batch class)')
    parser.add_argument('-qos', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_QOS, default=NeshCluster_Constants.DEFAULT_QOS, help='Quality of service on the Nesh-Cluster')
    parser.add_argument('-nodes', nargs='?', type=int, const=NeshCluster_Constants.DEFAULT_NODES, default=NeshCluster_Constants.DEFAULT_NODES, help='Number of nodes for the job on the Nesh-Cluster')
    parser.add_argument('-memory', nargs='?', type=int, const=None, default=None, help='Memory in GB for the job on the Nesh-Cluster')
    parser.add_argument('-time', nargs='?', type=int, const=None, default=None, help='Time in hours for the job on the Nesh-Cluster')

    args = parser.parse_args()
    parameterIdList = args.parameterIds if len(args.parameterIdRange) != 2 else range(args.parameterIdRange[0], args.parameterIdRange[1])

    concentrationId = None if args.concentrationId is None else int(args.concentrationId)
    spinupTolerance = None if args.spinupTolerance is None else float(args.spinupTolerance)

    main(args.metos3dModel, parameterIdList=parameterIdList, timestepList=args.timesteps, concentrationId=concentrationId, toleranceList=args.tolerance, yearIntervalList=args.yearInterval, spinupTolerance=spinupTolerance, partition=args.partition, qos=args.qos, nodes=args.nodes, memory=args.memory, time=args.time)

