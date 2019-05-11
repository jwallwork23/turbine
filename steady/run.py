from thetis_adjoint import *
from adapt_utils import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-dwr_approach")
parser.add_argument("-adapt_field")
parser.add_argument("-restrict")
parser.add_argument("-desired_error")
parser.add_argument("-initial_mesh", help="Choose initial mesh from 'uniform', 'coarse', 'fine'")
parser.add_argument("-num_turbines", help="Choose number of turbines from 2 or 15.")
parser.add_argument("-outer_startit")
parser.add_argument("-outer_maxit")
args = parser.parse_args()

num_turbines = 2 if args.num_turbines is None else int(args.num_turbines)
assert num_turbines in (2, 15)
approach = 'fixed_mesh' if args.approach is None else args.approach
initial_mesh = 'uniform' if args.initial_mesh is None else args.initial_mesh
if initial_mesh == 'uniform':
    mesh = None  # TODO: parallel version for 15-turbine
elif initial_mesh =='coarse':
    mesh = Mesh('../coarse_{:d}_turbine.msh'.format(num_turbines))  # TODO: for 15-turbine
elif initial_mesh =='fine':
    mesh = Mesh('../fine_{:d}_turbine.msh'.format(num_turbines))
    raise NotImplementedError
op = Steady2TurbineOptions(approach=approach) if num_turbines == 2 else Steady15TurbineOptions(approach=approach)

if args.dwr_approach is not None:
    op.dwr_approach = args.dwr_approach
op.restrict = 'error' if args.restrict is None else args.restrict
op.desired_error = 1e-5 if args.desired_error is None else float(args.desired_error)
op.adapt_field = 'fluid_speed' if args.adapt_field is None else args.adapt_field
op.family = 'dg-cg'
print(op)

#tp = SteadyTurbineProblem(mesh=mesh, op=op)
#tp.solve()
#tp.solve_adjoint()
#tp.adapt_mesh()
#tp.plot()

#mo = MeshOptimisation(SteadyTurbineProblem, op, mesh)
#mo.iterate()

ol = OuterLoop(SteadyTurbineProblem, op, mesh)
#ol.scale_to_convergence()
ol.outer_startit = 0 if args.outer_startit is None else int(args.outer_startit)
ol.outer_maxit = 4 if args.outer_maxit is None else int(args.outer_maxit)
ol.objective_rtol = 0.001  # TODO: temp
ol.desired_error_loop()
