#!/usr/bin/env python
# -*- coding: utf8 -*

import logging

from decreasingTimesteps.DecreasingTimestepsBackup import DecreasingTimestepsBackup


def main(backup=False, remove=True):
    """
    Create the backup of the simulation runs using the decreasing timesteps
    """
    logfile = 'DecreasingTimestepsBackup.log' if backup else 'DecreasingTimestepsRemove.log'
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=logfile, filemode='a', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    for parameterId in range(2, 101):
        dtBackup = DecreasingTimestepsBackup(parameterId = parameterId)
        if backup:
            dtBackup.backup()
        if remove:
            dtBackup.remove()


if __name__ == '__main__':
    main()

