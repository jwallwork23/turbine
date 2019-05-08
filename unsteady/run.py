from thetis_adjoint import *
from adapt_utils import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-num_turbines")
args = parser.parse_args()

num_turbines = 2 if args.num_turbines is None else int(args.num_turbines)
approach = 'fixed_mesh' if args.approach is None else args.approach

if num_turbines == 2:
    op = Unsteady2TurbineOptions(approach=approach)
    mesh = Mesh('../fine_2_turbine.msh')  # TODO: coarse option
elif num_turbines == 15:
    op = Unsteady15TurbineOptions(approach=approach)
    mesh = Mesh('../fine_15_turbine.msh')
else:
    raise NotImplementedError
op.family = 'dg-cg'
tp = UnsteadyTurbineProblem(op=op, mesh=mesh)
tp.solve()
