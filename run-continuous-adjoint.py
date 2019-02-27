from firedrake import *
from thetis import *
import math

from turbine.options import TurbineOptions


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


def solve_turbine_continuous_adjoint(mesh2d, op=TurbineOptions(approach='AdjointOnly')):
    """
    Solve steady state shallow water equations on a given mesh.

    :arg mesh2d: mesh upon which to solve the shallow water problem.
    :param op: `AdaptOptions` parameter class.
    :return: Thetis `FlowSolver2d` object containing solution fields; objective functional, as computed on `mesh2d`.
    """
    # if we solve with PressureProjectionPicard (theta=1.0) it seems to converge (power output to 7
    # digits) in roughly 800 timesteps of 20s with SteadyState we only do 1 timestep (t_end should
    # be slightly smaller than timestep to achieve this)
    t_end = 0.9*op.dt

    H = 40  # water depth
    H_const = Constant(H)

    # turbine parameters:
    D = 18                       # turbine diameter
    C_T = op.thrust_coefficient  # thrust coefficient

    # correction to account for the fact that the thrust coefficient is based on an upstream velocity
    # whereas we are using a depth averaged at-the-turbine velocity (see Kramer and Piggott 2016,
    # eq. (15))
    A_T = math.pi*(D/2)**2
    correction = 4/(1+math.sqrt(1-A_T/(H*D)))**2
    # NOTE, that we're not yet correcting power output here, so that will be overestimated

    # create solver and set options
    solver_obj = solver2d.FlowSolver2d(mesh2d, H_const)
    options = solver_obj.options
    options.timestep = op.dt
    options.simulation_export_time = op.dt
    options.simulation_end_time = t_end
    options.output_directory = op.directory()
    options.check_volume_conservation_2d = True
    options.use_grad_div_viscosity_term = op.symmetric_viscosity
    options.element_family = op.family
    options.timestepper_type = 'SteadyState'
    options.timestepper_options.solver_parameters['pc_factor_mat_solver_type'] = 'mumps'
    options.timestepper_options.solver_parameters['snes_monitor'] = True
    print("Using solver parameters {:s}".format(str(options.timestepper_options.solver_parameters)))
    #options.timestepper_options.implicitness_theta = 1.0
    options.horizontal_viscosity = Constant(op.viscosity)
    options.quadratic_drag_coefficient = Constant(op.drag_coefficient)
    options.use_lax_friedrichs_velocity = False  # TODO
    options.use_grad_depth_viscosity_term = False

    # assign boundary conditions
    left_tag = 1
    right_tag = 2
    top_bottom_tag = 3
    # noslip_bc = {'uv': Constant((0., 0.))}
    freeslip_bc = {'un': Constant(0.)}
    solver_obj.bnd_functions['shallow_water'] = {
        left_tag: {'uv': Constant((3., 0.))},
        # right_tag: {'un': Constant(3.), 'elev': Constant(0.)}
        right_tag: {'elev': Constant(0.)},
        # top_bottom_tag: noslip_bc,
        top_bottom_tag: freeslip_bc,
    }

    # Use bump, rather than indicator, function
    op.region_of_interest = [(xt1, yt1, D/2), (xt2, yt2, D/2)]
    turbine_density = op.bump(mesh2d, scale=len(op.region_of_interest)/assemble(op.bump(mesh2d)*dx))
    File(op.directory()+'Bump.pvd').write(turbine_density)

    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = turbine_density
    farm_options.turbine_options.diameter = D
    farm_options.turbine_options.thrust_coefficient = C_T * correction
    # turbine drag is applied everywhere (where the turbine density isn't zero)
    options.tidal_turbine_farms["everywhere"] = farm_options

    # callback that computes average power
    cb = turbines.TurbineFunctionalCallback(solver_obj)
    solver_obj.add_callback(cb, 'timestep')

    solver_obj.assign_initial_conditions(uv=as_vector((3.0, 0.0)))
    solver_obj.iterate()
    J = cb.average_power
    print("Average power: {p:.4e}".format(p=J))

    # Finite element spaces  # TODO: Use DG formulation for consistency
    P1_vec = VectorFunctionSpace(mesh2d, "CG", 1)
    P2_vec = VectorFunctionSpace(mesh2d, "CG", 2)
    P1 = FunctionSpace(mesh2d, "CG", 1)
    Taylor_Hood = P2_vec * P1

    # Trial and test functions
    z, zeta = TrialFunctions(Taylor_Hood)
    psi, phi = TestFunctions(Taylor_Hood)

    # Compute source term for adjoint equation
    u, eta = solver_obj.fields.solution_2d.split()
    C_D = C_T*correction*A_T/2.*turbine_density
    adjoint_source = 3.*C_D*u*sqrt(inner(u,u))
    adjoint_source_p1 = project(adjoint_source, P1_vec)
    adjoint_source_p1.rename("Source term for adjoint equation")
    File(op.directory() + 'AdjointSource2d.pvd').write(adjoint_source_p1)

    # Constants etc
    g = physical_constants['g_grav']
    nu = Constant(op.viscosity)
    C_B = op.drag_coefficient
    C = (C_D + C_B)/(eta+H_const)
    n = FacetNormal(mesh2d)

    # LHS contributions from adjoint momentum equation
    a = inner(dot(transpose(grad(u)), z), psi)*dx
    a += -div(u)*inner(z, psi)*dx
    a += inner(z, grad(dot(u, psi)))*dx
    a += nu*inner(grad(z), grad(psi))*dx
    a += zeta*div((eta + H_const)*psi)*dx
    a += C*(sqrt(inner(u, u))*inner(z, psi) + inner(u, z)*inner(u, psi)/sqrt(inner(u, u)))*dx
    a += -inner(u, psi)*dot(z, n)*ds(1)
    a += -inner(u, psi)*dot(z, n)*ds(3)
    a += -zeta*(eta+H_const)*dot(psi, n)*ds(1)
    a += -zeta*(eta+H_const)*dot(psi, n)*ds(3)
    a += -nu*inner(psi, dot(nabla_grad(z), n))*ds(1)
    a += -nu*inner(psi, dot(nabla_grad(z), n))*ds(3)

    # LHS contributions from adjoint continuity equation
    a += g*inner(z, grad(phi))*dx
    a += zeta*div(phi*u)*dx
    a += C/(eta+H_const)*sqrt(inner(u, u))*inner(u, z)*phi*dx
    a += -g*phi*dot(z, n)*ds(2)     # FIXME: Not sure here 
    #a += -zeta*phi*dot(u, n)*ds(1)  # FIXME: Not sure here
    a += -zeta*phi*dot(u, n)*ds(2)  # FIXME: Not sure here

    # RHS of adjoint equation
    L = inner(adjoint_source, psi)*dx

    # Solve adjoint system
    lam = Function(Taylor_Hood)
    z, zeta = lam.split()
    params = {
              'mat_type': 'aij',
              'pc_type': 'lu',
              'ksp_monitor': True,
              'ksp_converged_reason': True,
             }
    #bc = None
    bc = DirichletBC(Taylor_Hood.sub(0), 0, 'on_boundary')  # FIXME: Not apparent in discrete adjoint
    solve(a == L, lam, bcs=bc, solver_parameters=params)
    File(op.directory() + "AdjointSolution2d.pvd").write(z, zeta)


if __name__ == "__main__":

    mesh2d = Mesh('channel.msh')
    solve_turbine_continuous_adjoint(mesh2d)
