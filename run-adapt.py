from thetis_adjoint import *
from adapt_utils import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-dwr_approach", help="Choose DWR approach")
args = parser.parse_args()

# read global variables defining turbines from geo file
geo = open('channel.geo', 'r')
W = float(geo.readline().replace(';', '=').split('=')[1])
D = float(geo.readline().replace(';', '=').split('=')[1])
xt1 = float(geo.readline().replace(';', '=').split('=')[1])
xt2 = float(geo.readline().replace(';', '=').split('=')[1])
dx1 = float(geo.readline().replace(';', '=').split('=')[1])
dx2 = float(geo.readline().replace(';', '=').split('=')[1])
L = float(geo.readline().replace(';', '=').split('=')[1])
geo.close()
yt1=W/2
yt2=W/2

#tp = SteadyTurbineProblem(mesh=Mesh('channel.msh'), approach=args.approach)
#tp.solve()
#tp.solve_discrete_adjoint()
#tp.adapt_mesh()
#tp.plot()

op = TwoTurbineOptions(approach=args.approach)
if args.dwr_approach is not None:
    op.dwr_approach = args.dwr_approach
#op.desired_error = 1e-5
op.desired_error = 1e-4
#op.max_anisotropy = 50.
print(op)
mo = MeshOptimisation(SteadyTurbineProblem,
                      op=op,
                      mesh=Mesh('channel.msh'),
                      stab='lax_friedrichs',
                      approach=args.approach)
mo.iterate()

#ol = OuterLoop(SteadyTurbineProblem,
#               Mesh('channel.msh'),
#               approach=args.approach)
#ol.scale_to_convergence()
