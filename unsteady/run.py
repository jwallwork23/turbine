from thetis_adjoint import *
from adapt_utils import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-initial_mesh", help="Choose initial mesh from 'uniform', 'coarse', 'fine'")
parser.add_argument("-num_turbines")
args = parser.parse_args()

# initial mesh
initial_mesh = 'uniform' if args.initial_mesh is None else args.initial_mesh
if initial_mesh == 'uniform':
    mesh = None  # TODO: parallel version for 15-turbine
elif initial_mesh =='coarse':
    mesh = Mesh('../coarse_{:d}_turbine.msh'.format(num_turbines))  # TODO: for 15-turbine
elif initial_mesh =='fine':
    mesh = Mesh('../fine_{:d}_turbine.msh'.format(num_turbines))
    raise NotImplementedError

# parameters
num_turbines = 2 if args.num_turbines is None else int(args.num_turbines)
approach = 'fixed_mesh' if args.approach is None else args.approach
op = Steady2TurbineOptions(approach=approach) if num_turbines == 2 else Steady15TurbineOptions(approach=approach)
op.family = 'dg-cg'
op.end_time = op.T_tide

# solve
# TODO: store adjoint data
tp = UnsteadyTurbineProblem(op=op, mesh=mesh)
tp.solve()
