#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements a problem configuration
"""

import numpy as np
import configparser

class Configuration():
    """
    Defines the config
    """

    def loadconfig(self,path):
        """
        Loads the config
        :param path: Path to the config file
        """
        config = configparser.ConfigParser()
        config.read(path)
        self.poissonratio = float(config['Material']['PoissonRatio'])
        self.emodulus = float(config['Material']['E_Modulus'])
        self.density = float(config['Material']['Density'])
