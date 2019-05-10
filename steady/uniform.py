from thetis_adjoint import *
from adapt_utils import *
import argparse


# initialise
parser = argparse.ArgumentParser()
parser.add_argument("-maxit", help="Maximum number of mesh iterations.")
parser.add_argument("-element_rtol", help="Relative tolerance for convergence in element count.")
parser.add_argument("-objective_rtol", help="Relative tolerance for convergence in objective.")
args = parser.parse_args()
maxit = 4 if args.maxit is None else int(args.maxit)
element_rtol = 0.001 if args.element_rtol is None else float(args.element_rtol)
objective_rtol = 0.0001 if args.objective_rtol is None else float(args.objective_rtol)

# optimise
mo = MeshOptimisation(SteadyTurbineProblem, Steady2TurbineOptions(approach='uniform'))
mo.maxit=maxit
mo.element_rtol=element_rtol
mo.objective_rtol=objective_rtol
mo.iterate()
print('Done!')

