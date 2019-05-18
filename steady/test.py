from thetis_adjoint import *
from adapt_utils import *

num_turbines = 2
approach = 'dwr'
initial_mesh = 'coarse'
if initial_mesh == 'uniform':
    mesh = None  # TODO: parallel version for 15-turbine
elif initial_mesh =='coarse':
    mesh = Mesh('../coarse_{:d}_turbine.msh'.format(num_turbines))  # TODO: for 15-turbine
elif initial_mesh =='fine':
    mesh = Mesh('../fine_{:d}_turbine.msh'.format(num_turbines))
    raise NotImplementedError
op = Steady2TurbineOptions(approach=approach) if num_turbines == 2 else Steady15TurbineOptions(approach=approach)

# FIXME
#if initial_mesh == 'uniform':
#    op.boundary_conditions[4] = op.boundary_conditions[3]

op.restrict = 'target'
#op.restrict = 'p_norm'
#op.target = 1e+04
op.target = 1e+02
op.adapt_field = 'fluid_speed'
op.dwr_approach = 'cell_only'
op.family = 'dg-cg'
print(op)

#tp = SteadyTurbineProblem(mesh=mesh, op=op)
#tp.solve()
#tp.solve_adjoint()
#tp.adapt_mesh()
#tp.plot()

mo = MeshOptimisation(SteadyTurbineProblem, op, mesh)
mo.iterate()

#ol = OuterLoop(SteadyTurbineProblem, op, mesh)
##ol.scale_to_convergence()
#ol.outer_startit = 0
#ol.outer_maxit = 4
#ol.objective_rtol = 0.001  # TODO: temp
#ol.desired_error_loop()
