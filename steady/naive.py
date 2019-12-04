from thetis import *
from firedrake.petsc import PETSc
from adapt_utils.turbine.options import *
from adapt_utils.turbine.solver import SteadyTurbineProblem
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.adaptation import iso_P2
from adapt_utils.adapt.p0_metric import *
from adapt_utils.turbine.options import default_params
from pyop2.profiling import timed_stage
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-approach')
parser.add_argument('-target')
parser.add_argument('-offset')
parser.add_argument('-adapt_field')
args = parser.parse_args()

op2.init(log_level=INFO)

# Set parameters
approach = 'carpio' if args.approach is None else args.approach
num_adapt = 35  # Maximum iterations
level = 'xcoarse'
offset = False if args.offset is None else bool(args.offset)

label = level + '_2'
if offset:
    label += '_offset'
op = Steady2TurbineOffsetOptions(approach) if offset else Steady2TurbineOptions(approach)
op.timestepper = 'SteadyState'
op.target = 1000 if args.target is None else float(args.target)
op.family = 'dg-cg'  # NOTE: dg-cg seems to work better with various adapt_field choices
op.adapt_field = 'all_int' if args.adapt_field is None else args.adapt_field
op.normalisation = 'complexity'
# op.normalisation = 'error'
op.convergence_rate = 1
op.norm_order = None
op.num_adapt = num_adapt
sol = None

tol = 0.002
qoi_rtol = tol
element_rtol = tol
estimator_rtol = tol

# Initial mesh is important!!!
if level == 'uniform':
    mesh = op.default_mesh
else:
    mesh = Mesh(os.path.join('..', '{:s}_turbine.msh'.format(label)))
print("Number of elements: {:d}".format(mesh.num_cells()))

for i in range(op.num_adapt):
    print("Step {:d}".format(i))

    if not approach in ('fixed_mesh', 'uniform') or 'hessian' in approach:
        print("Solving in enriched space")
        #op.degree = 2
        enriched_mesh = Mesh(iso_P2(mesh).coordinates)
        tp_ho = SteadyTurbineProblem(mesh=enriched_mesh, discrete_adjoint=True, op=op, prev_solution=sol)
        tp_ho.solve()
        if approach != 'uniform' or 'hessian' in approach:
            tp_ho.solve_adjoint()

        print("Solving in base space")
    #op.degree = 1
    tp = SteadyTurbineProblem(mesh=mesh, discrete_adjoint=True, op=op, prev_solution=sol)
    tp.solve()
    print("Quantity of interest: {:.4e}".format(tp.qoi))
    if approach == 'fixed_mesh':
        break

    if i > 0 and np.abs(tp.qoi - qoi_old) < qoi_rtol*qoi_old:
        print("Number of elements: ", tp.mesh.num_cells())
        print("Number of dofs: ", sum(tp.V.dof_count))
        print("Converged quantity of interest!")
        break
    if i > 0 and np.abs(tp.mesh.num_cells() - num_cells_old) < element_rtol*num_cells_old:
        print("Number of elements: ", tp.mesh.num_cells())
        print("Number of dofs: ", sum(tp.V.dof_count))
        print("Converged number of mesh elements!")
        break
    if i == num_adapt-1:
        print("Did not converge!")
        break

    if approach != 'uniform' or 'hessian' in approach:
        tp.solve_adjoint()

        # Take difference
        adj_u, adj_eta = tp.adjoint_solution.split()
        adj_proj = Function(tp_ho.V)
        adj_proj_u, adj_proj_eta = adj_proj.split()
        adj_proj_u.project(adj_u)
        adj_proj_eta.project(adj_eta)
        adj_ho_u, adj_ho_eta = tp_ho.adjoint_solution.split()
        adj_ho_u -= adj_proj_u
        adj_ho_eta -= adj_proj_eta
        adj_u.project(adj_ho_u)
        adj_eta.project(adj_ho_eta)


    tp.indicate_error()
    print("Error estimator: {:.4e}".format(tp.estimators['dwr']))
    if i > 0 and np.abs(tp.estimators['dwr'] - estimator_old) < estimator_rtol*estimator_old:
        print("Number of elements: ", tp.mesh.num_cells())
        print("Number of dofs: ", sum(tp.V.dof_count))
        print("Converged error estimator!")
        break

    qoi_old = tp.qoi
    num_cells_old = tp.mesh.num_cells()
    estimator_old = tp.estimators['dwr']

    # Adapt mesh
    with timed_stage('mesh adapt {:d}'.format(i)):
        tp.adapt_mesh()
    mesh = tp.mesh
    sol = tp.solution

    # Save plex to HDF5
    # viewer = PETSc.Viewer().createHDF5('anisotropic_plex.h5', 'w')
    # viewer(mesh._plex)

print('\n'+80*'#')
print('SUMMARY')
print(80*'#' + '\n')
print('Target:               {:.1f}'.format(op.target))
print("Number of elements:   {:d}".format(tp.mesh.num_cells()))
print("Number of dofs:       {:d}".format(sum(tp.V.dof_count)))
print("Quantity of interest: {:.4e}".format(tp.qoi))
print(80*'#')
