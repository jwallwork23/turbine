from thetis_adjoint import *
from adapt_utils import *

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-dwr_approach")
parser.add_argument("-adapt_field")
parser.add_argument("-normalisation")
parser.add_argument("-target")
parser.add_argument("-initial_mesh", help="Choose initial mesh from 'uniform', 'coarse', 'fine'")
parser.add_argument("-num_turbines", help="Choose number of turbines from 2 or 15.")
parser.add_argument("-outer_startit")
parser.add_argument("-outer_maxit")
parser.add_argument("-maxit")
parser.add_argument("-objective_rtol")
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
    label = initial_mesh + '_' + str(num_turbines)
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

if args.dwr_approach is not None:
    op.dwr_approach = args.dwr_approach
op.normalisation = 'complexity' if args.normalisation is None else args.normalisation
op.target = 1e+3 if args.target is None else float(args.target)
op.adapt_field = 'all' if args.adapt_field is None else args.adapt_field
op.family = 'dg-dg'
op.relax = False
op.convergence_rate = 1  # NOTE: This parameter seems to be very important!
op.qoi_rtol = 0.001
print(op)

#tp = SteadyTurbineProblem(mesh=mesh, op=op)
#tp.solve()
#tp.solve_adjoint()
#tp.adapt_mesh()
#tp.plot()

mo = MeshOptimisation(SteadyTurbineProblem, op, mesh)
mo.iterate()
exit(0)  # TODO: temp

ol = OuterLoop(SteadyTurbineProblem, op, mesh)
#ol.scale_to_convergence()
ol.outer_startit = 2 if args.outer_startit is None else int(args.outer_startit)
ol.outer_maxit = 7 if args.outer_maxit is None else int(args.outer_maxit)
ol.maxit = 35 if args.maxit is None else int(args.maxit)
if args.objective_rtol is not None:
    ol.objective_rtol = float(args.objective_rtol)
ol.desired_error_loop()
