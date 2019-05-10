from thetis_adjoint import *
from adapt_utils import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-dwr_approach")
parser.add_argument("-adapt_field")
parser.add_argument("-restrict")
parser.add_argument("-desired_error")
parser.add_argument("-num_turbines", help="Choose number of turbines from 2 or 15.")
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

if args.dwr_approach is not None:
    op.dwr_approach = args.dwr_approach
op.restrict = 'error' if args.restrict is None else args.restrict
op.desired_error = 1e-5 if args.desired_error is None else float(args.desired_error)
op.adapt_field = 'fluid_speed' if args.adapt_field is None else args.adapt_field
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
ol.desired_error_loop()
