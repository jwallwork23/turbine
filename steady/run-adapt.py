from thetis_adjoint import *
from adapt_utils import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-dwr_approach")
parser.add_argument("-num_turbines")
args = parser.parse_args()

num_turbines = 2 if args.num_turbines is None else int(args.num_turbines)
approach = 'fixed_mesh' if args.approach is None else args.approach
if num_turbines == 2:
    op = Steady2TurbineOptions(approach=approach)
    mesh = Mesh('../fine_2_turbine.msh')
elif num_turbines == 15:
    op = Steady15TurbineOptions(approach=approach)
    mesh = Mesh('../fine_15_turbine.msh')
else:
    raise NotImplementedError

#tp = SteadyTurbineProblem(mesh=mesh, op=op)
#tp.solve()
#tp.solve_adjoint()
#tp.adapt_mesh()
#tp.plot()

if args.dwr_approach is not None:
    op.dwr_approach = args.dwr_approach
#op.desired_error = 1e-5
op.desired_error = 1e-2
#op.max_anisotropy = 50.
print(op)
mo = MeshOptimisation(SteadyTurbineProblem, op, mesh)
mo.iterate()

#ol = OuterLoop(SteadyTurbineProblem, op, mesh)
#ol.scale_to_convergence()
