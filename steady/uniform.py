from thetis import *
from adapt_utils import *
import argparse


# initialise
parser = argparse.ArgumentParser()
parser.add_argument("-startit", help="Starting iterations.")
parser.add_argument("-maxit", help="Maximum number of mesh iterations.")
parser.add_argument("-objective_rtol", help="Relative tolerance for convergence in objective.")
parser.add_argument("-initial_mesh", help="Choose initial mesh from 'uniform', 'coarse', 'fine'")
parser.add_argument("-num_turbines")
args = parser.parse_args()
startit = 0 if args.startit is None else int(args.startit)
maxit = 6 if args.maxit is None else int(args.maxit)
objective_rtol = 0.0001 if args.objective_rtol is None else float(args.objective_rtol)

num_turbines = 2 if args.num_turbines is None else int(args.num_turbines)
initial_mesh = 'uniform' if args.initial_mesh is None else args.initial_mesh
if initial_mesh == 'uniform':
    mesh = None  # TODO: parallel version for 15-turbine
elif initial_mesh =='coarse':
    mesh = Mesh('../coarse_{:d}_turbine.msh'.format(num_turbines))  # TODO: for 15-turbine
elif initial_mesh =='fine':
    mesh = Mesh('../fine_{:d}_turbine.msh'.format(num_turbines))
    raise NotImplementedError

# optimise
op = Steady2TurbineOptions(approach='uniform') if num_turbines == 2 else Steady15TurbineOptions(approach='uniform')
op.family = 'dg-cg'
mo = MeshOptimisation(SteadyTurbineProblem, op, mesh=mesh)
mo.startit = startit
mo.maxit = maxit
mo.objective_rtol = objective_rtol
mo.iterate()
print('Done!')

