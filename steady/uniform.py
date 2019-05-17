from thetis import *
from firedrake.petsc import PETSc
from adapt_utils import *
import argparse


# initialise
parser = argparse.ArgumentParser()
parser.add_argument("-resolution")
#parser.add_argument("-initial_mesh", help="Choose initial mesh from 'uniform', 'coarse', 'fine'")
parser.add_argument("-num_turbines")
args = parser.parse_args()

num_turbines = 2 if args.num_turbines is None else int(args.num_turbines)
# TODO: MeshHierarchy for coarse/fine
#initial_mesh = 'uniform' if args.initial_mesh is None else args.initial_mesh
#if initial_mesh == 'uniform':
#    mesh = None  # TODO: parallel version for 15-turbine
#elif initial_mesh =='coarse':
#    mesh = Mesh('../coarse_{:d}_turbine.msh'.format(num_turbines))  # TODO: for 15-turbine
#elif initial_mesh =='fine':
#    mesh = Mesh('../fine_{:d}_turbine.msh'.format(num_turbines))
#else:
#    raise NotImplementedError
n = 0 if args.resolution is None else float(args.resolution)
mesh = RectangleMesh(100*2**n, 20*2**n, 1000, 200)

# optimise
op = Steady2TurbineOptions(approach='uniform') if num_turbines == 2 else Steady15TurbineOptions(approach='uniform')
op.family = 'dg-cg'
tp = SteadyTurbineProblem(mesh, op)
tp.solve()
J = tp.objective_functional()
PETSc.Sys.Print("Objective value on mesh with %d elements: %.4e" % (mesh.num_cells(), J), comm=COMM_WORLD)
