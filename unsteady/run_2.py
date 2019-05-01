from thetis_adjoint import *
from adapt_utils import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
args = parser.parse_args()

op = Unsteady2TurbineOptions(approach='fixed_mesh' if args.approach is None else args.approach)
op.family = 'dg-cg'
tp = UnsteadyTurbineProblem(op=op, mesh=Mesh('../coarse_2_turbine.msh'))
tp.solve()
