#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA waypoint follower assessment client script.

A controller assessment to follow a given trajectory, where the trajectory
can be defined using way-points.

STARTING in a moment...
"""
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import controller2d
import configparser

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import live_plotter as lv   # Custom live plotting library
from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils

client = carla.Client('localhost', 2000)
world = client.get_world()
map = world.get_map()
print('Mapa: ' + map)
spawn_points = world.get_map().get_spawn_points()
print('Puntos de aparicion: '+ spawn_points)
