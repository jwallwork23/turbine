from thetis_adjoint import *
from adapt_utils import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-adapt_field")
parser.add_argument("-initial_mesh", help="Choose initial mesh from 'uniform', 'coarse', 'fine'")
parser.add_argument("-num_turbines")
parser.add_argument("-desired_error")
parser.add_argument("-solve_adjoint")  # TODO: store adjoint for each mesh
args = parser.parse_args()

# initial mesh
num_turbines = 2 if args.num_turbines is None else int(args.num_turbines)
initial_mesh = 'uniform' if args.initial_mesh is None else args.initial_mesh
if initial_mesh == 'uniform':
    mesh = None  # TODO: parallel version for 15-turbine
elif initial_mesh =='coarse':
    mesh = Mesh('../coarse_{:d}_turbine.msh'.format(num_turbines))  # TODO: for 15-turbine
elif initial_mesh =='fine':
    mesh = Mesh('../fine_{:d}_turbine.msh'.format(num_turbines))
    raise NotImplementedError

# parameters
approach = 'fixed_mesh' if args.approach is None else args.approach
op = Unsteady2TurbineOptions(approach=approach) if num_turbines == 2 else Unsteady15TurbineOptions(approach=approach)
op.family = 'dg-cg'
op.adapt_field = 'fluid_speed' if args.adapt_field is None else args.adapt_field
op.end_time = op.T_tide
op.desired_error = 1e-2 if args.desired_error is None else float(args.desired_error)

adj = False if args.solve_adjoint is None else bool(args.solve_adjoint)

# solve
if adj:
    op_ = Unsteady2TurbineOptions() if num_turbines == 2 else Unsteady15TurbineOptions()
    op_.family = 'dg-cg'
    tp = UnsteadyTurbineProblem(op=op_, mesh=mesh)
    tp.solve()
    tp.solve_adjoint()
tp = UnsteadyTurbineProblem(op=op, mesh=mesh)
tp.solve()
