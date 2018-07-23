# simple set up with two turbines
from thetis import *
# from thetis_adjoint import *
# import pyadjoint
# from fenics_adjoint.solving import SolveBlock                                       # For extracting adjoint solutions
import math
# op2.init(log_level=INFO)

from utils.adaptivity import *
from utils.interpolation import mixed_pair_interp
from utils.options import TurbineOptions


def solve_turbine(mesh2d):
    # if we solve with PressureProjectionPicard (theta=1.0) it seems to converge (power output to 7 digits) in roughly
    # 800 timesteps of 20s
    # with SteadyState we only do 1 timestep (t_end should be slightly smaller than timestep to achieve this)
    timestep = 20
    t_end = 0.9*timestep

    H = 40  # water depth

    # turbine parameters:
    D = 18  # turbine diameter
    C_T = 0.8  # thrust coefficient

    # correction to account for the fact that the thrust coefficient is based on an upstream velocity
    # whereas we are using a depth averaged at-the-turbine velocity (see Kramer and Piggott 2016, eq. (15))
    A_T = math.pi*(D/2)**2
    correction = 4/(1+math.sqrt(1-A_T/(H*D)))**2
    # NOTE, that we're not yet correcting power output here, so that will be overestimated

    # create solver and set options
    solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(H))
    options = solver_obj.options
    options.timestep = timestep
    options.simulation_export_time = timestep
    options.simulation_end_time = t_end
    options.output_directory = 'outputs'
    options.check_volume_conservation_2d = True
    options.element_family = 'dg-dg'
    options.timestepper_type = 'SteadyState'
    options.timestepper_options.solver_parameters['pc_factor_mat_solver_type'] = 'mumps'
    options.timestepper_options.solver_parameters['snes_monitor'] = True
    #options.timestepper_options.implicitness_theta = 1.0
    options.horizontal_viscosity = Constant(0.1)
    options.quadratic_drag_coefficient = Constant(0.0025)

    # assign boundary conditions
    left_tag = 1
    right_tag = 2
    # noslip currently doesn't work (vector Constants are broken in firedrake_adjoint)
    freeslip_bc = {'un': Constant(0.0)}
    solver_obj.bnd_functions['shallow_water'] = {
        left_tag: {'uv': Constant((3., 0.))},
        right_tag: {'un': Constant(3.), 'elev': Constant(0.)}
    }

    # we've meshed the turbines as DxD squares, so we can treat it
    # as turbine "farm"s with turbine density of 1 turbine per D^2 area
    turbine_density = Constant(1.0/D**2, domain=mesh2d)
    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = turbine_density
    farm_options.turbine_options.diameter = D
    farm_options.turbine_options.thrust_coefficient = C_T*correction
    # assign ID 2,3 with the "farm"
    options.tidal_turbine_farms[2] = farm_options
    options.tidal_turbine_farms[3] = farm_options


    # callback that computes average power
    cb = turbines.TurbineFunctionalCallback(solver_obj)
    solver_obj.add_callback(cb, 'timestep')

    solver_obj.assign_initial_conditions(uv=as_vector((3.0, 0.0)))
    solver_obj.iterate()
    print("Average power = {p:.4e}".format(p=cb.average_power))

    return solver_obj.fields.solution_2d


def mesh_adapt(solution, op=TurbineOptions()):
    mesh2d = solution.function_space().mesh()

    op.target_vertices = mesh2d.num_vertices() * op.rescaling
    uv_2d, elev_2d = solution.split()
    P1 = FunctionSpace(mesh2d, "CG", 1)

    if op.approach == 'HessianBased'
        if op.adapt_field != 'f':       # metric for fluid speed
            spd = Function(P1)
            spd.interpolate(sqrt(inner(uv_2d, uv_2d)))
            M = steady_metric(spd, op=op)
        if op.adapt_field != 's':       # metric for free surface
            M2 = steady_metric(elev_2d, op=op)
        if op.adapt_field == 'b':       # intersect metrics for fluid speed and free surface
            M = metric_intersection(M, M2)
        elif op.adapt_field == 'f':
            M = M2

    elif op.approach in ('DWP', 'DWR'):
        epsilon = Function(P1, name='error_2d')

        # load error indicators
        with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_00000', mode=FILE_READ) as le:
            le.load(epsilon)
            le.close()
        estimate = Function(P1).assign(interp(mesh2d, epsilon))
        print("#### DEBUG: error estimator norm = %.4e" % norm(estimate))

        # compute metric field
        M = isotropic_metric(estimate, invert=False, op=op)
        if op.gradate:
            bdy = 'on_boundary'  # use boundary tags to gradate to individual boundaries
            H0 = Function(P1).interpolate(CellSize(mesh2d))
            M_ = isotropic_metric(interp(mesh2d, H0), bdy=bdy, op=op)  # Initial boundary metric
            M = metric_intersection(M, M_, bdy=bdy)
            M = gradate_metric(M, op=op)
    mesh2d = AnisotropicAdaptation(mesh2d, M).adapted_mesh

    return mesh2d


def get_error_estimators(mesh2d, op=TurbineOptions()):
    # if we solve with PressureProjectionPicard (theta=1.0) it seems to converge (power output to 7 digits) in roughly
    # 800 timesteps of 20s
    # with SteadyState we only do 1 timestep (t_end should be slightly smaller than timestep to achieve this)
    timestep = 20
    t_end = 0.9*timestep

    H = 40  # water depth
    H_const = Constant(H)

    # turbine parameters:
    D = 18  # turbine diameter
    C_T = 0.8  # thrust coefficient

    # correction to account for the fact that the thrust coefficient is based on an upstream velocity
    # whereas we are using a depth averaged at-the-turbine velocity (see Kramer and Piggott 2016, eq. (15))
    A_T = math.pi*(D/2)**2
    correction = 4/(1+math.sqrt(1-A_T/(H*D)))**2
    # NOTE, that we're not yet correcting power output here, so that will be overestimated

    # create solver and set options
    solver_obj = solver2d.FlowSolver2d(mesh2d, H_const)
    options = solver_obj.options
    options.timestep = timestep
    options.simulation_export_time = timestep
    options.simulation_end_time = t_end
    options.output_directory = 'outputs'
    options.check_volume_conservation_2d = True
    options.element_family = 'dg-dg'
    options.timestepper_type = 'SteadyState'
    options.timestepper_options.solver_parameters['pc_factor_mat_solver_type'] = 'mumps'
    options.timestepper_options.solver_parameters['snes_monitor'] = True
    #options.timestepper_options.implicitness_theta = 1.0
    options.horizontal_viscosity = Constant(0.1)
    options.quadratic_drag_coefficient = Constant(0.0025)

    # assign boundary conditions
    left_tag = 1
    right_tag = 2
    # noslip currently doesn't work (vector Constants are broken in firedrake_adjoint)
    freeslip_bc = {'un': Constant(0.0)}
    solver_obj.bnd_functions['shallow_water'] = {
        left_tag: {'uv': Constant((3., 0.))},
        right_tag: {'un': Constant(3.), 'elev': Constant(0.)}
    }

    # we've meshed the turbines as DxD squares, so we can treat it
    # as turbine "farm"s with turbine density of 1 turbine per D^2 area
    turbine_density = Constant(1.0/D**2, domain=mesh2d)
    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = turbine_density
    farm_options.turbine_options.diameter = D
    farm_options.turbine_options.thrust_coefficient = C_T*correction
    # assign ID 2,3 with the "farm"
    options.tidal_turbine_farms[2] = farm_options
    options.tidal_turbine_farms[3] = farm_options


    # callback that computes average power
    cb1 = turbines.TurbineFunctionalCallback(solver_obj)
    if op.order_increase:
        cb2 = callback.InteriorResidualCallback(solver_obj, export_to_hdf5=True)
    else:
        cb2 = callback.ExplicitErrorCallback(solver_obj, export_to_hdf5=True)
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.add_callback(cb2, 'export')

    solver_obj.assign_initial_conditions(uv=as_vector((3.0, 0.0)))
    solver_obj.iterate()
    J = cb1.average_power
    print("Average power = {p:.4e}".format(p=J))

    compute_gradient(J, Control(H_const))
    tape = get_working_tape()
    solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
    N = len(solve_blocks)
    try:
        assert N == 1
    except:
        print("#### DEBUG: Number of solves = {:d}".format(N))

    # create function spaces and get primal and dual solutions
    V = op.mixed_space(mesh2d)
    dual = Function(V)
    dual.assign(solve_blocks[0].adj_sol)
    dual_u, dual_e = dual.split()
    q = solver_obj.fields.solution_2d
    higher_order_dual = Function(op.mixed_space(mesh2d, enrich=True))
    higher_order_dual_u, higher_order_dual_e = higher_order_dual.split()
    P1 = FunctionSpace(mesh2d, "CG", 1)
    epsilon = Function(P1)

    # form error indicator
    if op.approach == "DWP":
        epsilon.interpolate(q, dual)
    else:
        tag = 'InteriorResidual2d_' if op.order_increase else 'ExplicitError2d_'
        with DumbCheckpoint(op.directory() + 'hdf5/' + tag + '00000', mode=FILE_READ) as lr:
            if op.order_increase:
                residual_2d = Function(V)
                res_u, res_e = residual_2d.split()
                lr.load(res_u, name="momentum residual")
                lr.load(res_e, name="continuity residual")
                higher_order_dual_u.interpolate(dual_u)
                higher_order_dual_e.interpolate(dual_e)
                epsilon.interpolate(inner(res_u, higher_order_dual_u) + res_e * higher_order_dual_e)
            else:
                residual_2d = Function(P1)
                lr.load(residual_2d, name="explicit error")
                epsilon.interpolate(residual_2d * local_norm(dual))
            lr.close()
    epsilon = normalise_indicator(epsilon, op=op)
    epsilon.rename('error_2d')
    with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_00000', mode=FILE_CREATE) as se:
        se.store(epsilon)
        se.close()
    File(op.directory() + "ErrorIndicator2d.pvd").write(epsilon)

    return q


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="Choose adaptive approach from {'HessianBased', 'DWP', 'DWR'} (default 'FixedMesh')")
    parser.add_argument("-f", help="Choose field to adapt to from {'s', 'f', 'b'}, denoting speed, free surface and both, resp.")
    parser.add_argument("-n", help="Number of mesh adaptation steps")
    args = parser.parse_args()

    op = TurbineOptions()
    if args.a is not None:
        op.approach = args.a
    if args.f is not None:
        op.adapt_field = args.f
    n = int(args.n) if args.n is not None else 1

    mesh2d = Mesh('channel.msh')
    if op.approach == "FixedMesh":
        solve_turbine(mesh2d)
    else:
        for i in range(n):
            print("\n#### Solving on mesh {i:d}\n".format(i=i))
            sol = solve_turbine(mesh2d)
            if i < n-1:
                mesh2d = mesh_adapt(sol, op=op)
        sol = mixed_pair_interp(mesh2d, sol)
        uv, elev = sol.split()
        uv.rename('uv_2d')
        elev.rename('elev_2d')
        File('outputs/final_solution.pvd').write(uv, elev)
