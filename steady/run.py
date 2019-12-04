from thetis_adjoint import *
from adapt_utils import *

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-adapt_field")
parser.add_argument("-normalisation")
parser.add_argument("-target")
parser.add_argument("-initial_mesh", help="Choose initial mesh from 'uniform', 'coarse', 'fine'")
parser.add_argument("-num_turbines", help="Choose number of turbines from 2 or 15.")
parser.add_argument("-outer_startit")
parser.add_argument("-outer_maxit")
parser.add_argument("-maxit")
parser.add_argument("-qoi_rtol")
parser.add_argument("-convergence_rate")
parser.add_argument("-offset")
args = parser.parse_args()

num_turbines = 2 if args.num_turbines is None else int(args.num_turbines)
assert num_turbines in (1, 2, 15)
approach = 'fixed_mesh' if args.approach is None else args.approach
initial_mesh = 'xcoarse' if args.initial_mesh is None else args.initial_mesh
offset = False if args.offset is None else bool(args.offset)
try:
    if offset:
        assert num_turbines == 2
except:
    raise NotImplementedError  # TODO
if initial_mesh == 'uniform':
    mesh = None
else:
    label = '_'.join([initial_mesh, str(num_turbines)])
    if offset:
        label += '_offset'
    mesh = Mesh(os.path.join('..', label + '_turbine.msh'))
if num_turbines == 1:
    raise NotImplementedError
    op = Steady1TurbineOptions(approach)
elif num_turbines == 2:
    op = Steady2TurbineOffsetOptions(approach) if offset else Steady2TurbineOptions(approach)
else:
    Steady15TurbineOptions(approach)

op.normalisation = 'complexity' if args.normalisation is None else args.normalisation
op.target = 1e+3 if args.target is None else float(args.target)
op.adapt_field = 'inflow_and_elevation' if args.adapt_field is None else args.adapt_field
op.family = 'dg-dg'
op.relax = False
# NOTE: This parameter seems to be very important!
op.convergence_rate = 1 if args.convergence_rate is None else int(args.convergence_rate) 
print(op)

mo = MeshOptimisation(SteadyTurbineProblem, op, mesh)
mo.use_prev_sol = True
mo.qoi_rtol = 0.005 if args.qoi_rtol is None else float(args.qoi_rtol)
mo.iterate()
