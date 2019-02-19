from thetis import *
from thetis.configuration import *

import numpy as np

from adapt.options import AdaptOptions


__all__ = ["TurbineOptions"]


class TurbineOptions(AdaptOptions):
    name = 'Parameters for the 2 turbine problem'
    mode = 'Turbine'

    # Solver parameters
    dt = PositiveFloat(20.).tag(config=True)
    target_vertices = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    adapt_field = Unicode('fluid_speed', help="Adaptation field of interest, from {'fluid_speed', 'elevation', 'both'}.").tag(config=True)
    h_min = PositiveFloat(1e-4, help="Minimum element size").tag(config=True)
    h_max = PositiveFloat(10., help="Maximum element size").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    viscosity = NonNegativeFloat(1.).tag(config=True)
    drag_coefficient = NonNegativeFloat(0.0025).tag(config=True)
    north_south_bc = Unicode(None, allow_none=True, help="Set North and South boundary conditions, from {None, 'freeslip', 'noslip'}").tag(config=True)
