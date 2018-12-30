# simple set up with two turbines
from thetis_adjoint import *
from fenics_adjoint.solving import SolveBlock   # For extracting adjoint solutions
import math
# op2.init(log_level=INFO)

from utils.adaptivity import *
from utils.interpolation import interp
from utils.options import TurbineOptions

# read global variables defining turbines from geo file
geo = open('channel.geo', 'r')
W = float(geo.readline().replace(';', '=').split('=')[1])
D = float(geo.readline().replace(';', '=').split('=')[1])
xt1 = float(geo.readline().replace(';', '=').split('=')[1])
xt2 = float(geo.readline().replace(';', '=').split('=')[1])
geo.close()
yt1=W/2
yt2=W/2


def solve_turbine(mesh2d, op=TurbineOptions()):
    """
    Solve steady state shallow water equations on mesh `mesh2d`, using `AdaptOptions` parameter class `op`.

    :return: approximate solution tuple for steady state shallow water equations.
    """
    # if we solve with PressureProjectionPicard (theta=1.0) it seems to converge (power output to 7 digits) in roughly
    # 800 timesteps of 20s
    # with SteadyState we only do 1 timestep (t_end should be slightly smaller than timestep to achieve this)
    t_end = 0.9*op.dt

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
    options.timestep = op.dt
    options.simulation_export_time = op.dt
    options.simulation_end_time = t_end
    options.output_directory = op.directory()
    options.check_volume_conservation_2d = True
    options.element_family = op.family
    options.timestepper_type = 'SteadyState'
    options.timestepper_options.solver_parameters['pc_factor_mat_solver_type'] = 'mumps'
    options.timestepper_options.solver_parameters['snes_monitor'] = True
    print("Using solver parameters {:s}".format(str(options.timestepper_options.solver_parameters)))
    #options.timestepper_options.implicitness_theta = 1.0
    options.horizontal_viscosity = Constant(op.viscosity)
    options.quadratic_drag_coefficient = Constant(op.drag_coefficient)

    # assign boundary conditions
    left_tag = 1
    right_tag = 2
    # noslip currently doesn't work (vector Constants are broken in firedrake_adjoint)
    freeslip_bc = {'un': Constant(0.0)}
    solver_obj.bnd_functions['shallow_water'] = {
        left_tag: {'uv': Constant((3., 0.))},
        # right_tag: {'un': Constant(3.), 'elev': Constant(0.)}
        right_tag: {'elev': Constant(0.)}
    }

    # we haven't meshed the turbines with separate ids, so define a farm everywhere
    # and make it have a density of 1/D^2 inside the two DxD squares where the turbines are
    # and 0 outside
    P1DG = FunctionSpace(mesh2d, "DG", 0)
    x, y = SpatialCoordinate(mesh2d)
    turbine_density = Function(P1DG)    # note pyadjoint can't deal with coordinateless functions
    turbine_density.interpolate(conditional(
        Or(
            And(And(gt(x, xt1 - D / 2), lt(x, xt1 + D / 2)), And(gt(y, yt1 - D / 2), lt(y, yt1 + D / 2))),
            And(And(gt(x, xt2 - D / 2), lt(x, xt2 + D / 2)), And(gt(y, yt2 - D / 2), lt(y, yt2 + D / 2)))
        ), 1.0 / D ** 2, 0))
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
    print("Average power = {p:.4e}".format(p=cb.average_power))

    return solver_obj.fields.solution_2d


def get_error_estimators(mesh2d, op=TurbineOptions()):
    """
    Generate a posteriori error indicators on mesh `mesh2d` using `AdaptOptions` parameter class `op`.

    :return: approximate solution to steady state shallow water equations, a posteriori error estimate
    """
    # if we solve with PressureProjectionPicard (theta=1.0) it seems to converge (power output to 7 digits) in roughly
    # 800 timesteps of 20s
    # with SteadyState we only do 1 timestep (t_end should be slightly smaller than timestep to achieve this)
    t_end = 0.9*op.dt

    H = 40. # water depth
    H_const = Constant(H)

    # Vector constants are broken in pyadjoint, so use a vector function instead
    P1 = VectorFunctionSpace(mesh2d, 'CG', 1)
    inflow = Function(P1).interpolate(as_vector([3., 0.]))

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
    options.timestep = op.dt
    options.simulation_export_time = op.dt
    options.simulation_end_time = t_end
    options.output_directory = op.directory()
    options.check_volume_conservation_2d = True
    options.element_family = op.family
    options.timestepper_type = 'SteadyState'
    options.timestepper_options.solver_parameters['pc_factor_mat_solver_type'] = 'mumps'
    options.timestepper_options.solver_parameters['snes_monitor'] = True
    print("Using solver parameters {:s}".format(str(options.timestepper_options.solver_parameters)))
    #options.timestepper_options.implicitness_theta = 1.0
    options.horizontal_viscosity = Constant(op.viscosity)
    options.quadratic_drag_coefficient = Constant(op.drag_coefficient)

    # assign boundary conditions
    left_tag = 1
    right_tag = 2
    # noslip currently doesn't work (vector Constants are broken in firedrake_adjoint)
    freeslip_bc = {'un': Constant(0.0)}
    solver_obj.bnd_functions['shallow_water'] = {
        #left_tag: {'uv': Constant((3., 0.))},
        left_tag: {'uv': inflow},
        # right_tag: {'un': Constant(3.), 'elev': Constant(0.)}
        right_tag: {'elev': Constant(0.)}
    }

    # we haven't meshed the turbines with separate ids, so define a farm everywhere
    # and make it have a density of 1/D^2 inside the two DxD squares where the turbines are
    # and 0 outside
    P1DG = FunctionSpace(mesh2d, "DG", 0)
    x, y = SpatialCoordinate(mesh2d)
    turbine_density = Function(P1DG)  # note pyadjoint can't deal with coordinateless functions
    turbine_density.interpolate(conditional(
        Or(
            And(And(gt(x, xt1 - D / 2), lt(x, xt1 + D / 2)), And(gt(y, yt1 - D / 2), lt(y, yt1 + D / 2))),
            And(And(gt(x, xt2 - D / 2), lt(x, xt2 + D / 2)), And(gt(y, yt2 - D / 2), lt(y, yt2 + D / 2)))
        ), 1.0 / D ** 2, 0))
    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = turbine_density
    farm_options.turbine_options.diameter = D
    farm_options.turbine_options.thrust_coefficient = C_T * correction
    # turbine drag is applied everywhere (where the turbine density isn't zero)
    options.tidal_turbine_farms["everywhere"] = farm_options

    # callback that computes average power
    cb1 = turbines.TurbineFunctionalCallback(solver_obj)
    solver_obj.add_callback(cb1, 'timestep')
    if op.approach == "DWR":
        if op.order_increase:
            cb2 = callback.CellResidualCallback(solver_obj, export_to_hdf5=True, export_to_pvd=True)
            #cb3 = callback.EdgeResidualCallback(solver_obj, export_to_hdf5=True, export_to_pvd=True)
        else:
            cb2 = callback.ExplicitErrorCallback(solver_obj, export_to_hdf5=True, export_to_pvd=True)
        solver_obj.add_callback(cb2, 'export')

    solver_obj.assign_initial_conditions(uv=as_vector((3.0, 0.0)))
    solver_obj.iterate()
    J = cb1.average_power
    print("Average power = {:.4e}".format(J))

    compute_gradient(J, Control(H_const))
    tape = get_working_tape()
    solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock) and block.adj_sol is not None]
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
    dual_u.rename('adjoint_velocity_2d')
    dual_e.rename('adjoint_elev_2d')
    File(op.directory() + "AdjointSolution2d.pvd").write(dual_u, dual_e)
    q = solver_obj.fields.solution_2d
    P1 = FunctionSpace(mesh2d, "CG", 1)
    epsilon = Function(P1)

    # form error indicator
    if op.approach == "DWP":
        epsilon.interpolate(inner(q, dual))
    else:
        tag = 'CellResidual2d_' if op.order_increase else 'ExplicitError2d_'
        with DumbCheckpoint(op.directory() + 'hdf5/' + tag + '00001', mode=FILE_READ) as lr:
            if op.order_increase:
                residual_2d = Function(V)
                res_u, res_e = residual_2d.split()
                lr.load(res_u, name="momentum residual")
                lr.load(res_e, name="continuity residual")

                # TODO: Needs patchwise interpolation for proper implementation
                higher_order_dual = Function(op.mixed_space(mesh2d, enrich=True))
                higher_order_dual_u, higher_order_dual_e = higher_order_dual.split()
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
    File(op.directory() + "ErrorIndicator2d.pvd").write(epsilon)
    tape.clear_tape()

    return q, epsilon


def mesh_adapt(solution, error_indicator=None, op=TurbineOptions()):
    """
    Adapt mesh based on an error indicator or field of interest.

    :param solution: approximate solution of prognostic equations.
    :param error_indicator: optional error indicator upon which to adapt.
    :param op: `AdaptOptions` parameter class.
    :return: adapted mesh.
    """
    mesh2d = solution.function_space().mesh()
    op.target_vertices = mesh2d.num_vertices() * op.rescaling
    P1 = FunctionSpace(mesh2d, "CG", 1)

    if op.approach == 'HessianBased':
        uv_2d, elev_2d = solution.split()
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
        assert(error_indicator is not None)
        print("#### DEBUG: error estimator norm = %.4e" % norm(error_indicator))

        # compute metric field
        M = isotropic_metric(error_indicator, invert=False, op=op)
        if op.gradate:
            bdy = 'on_boundary'  # use boundary tags to gradate to individual boundaries
            H0 = Function(P1).interpolate(CellSize(mesh2d))
            M_ = isotropic_metric(interp(mesh2d, H0), bdy=bdy, op=op)  # Initial boundary metric
            M = metric_intersection(M, M_, bdy=bdy)
            M = gradate_metric(M, op=op)
    mesh2d = AnisotropicAdaptation(mesh2d, M).adapted_mesh
    print("Number of elements after mesh adaptation: {:d}".format(mesh2d.num_cells()))

    return mesh2d


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="Choose adaptive approach from {'HessianBased', 'DWP', 'DWR'} (default 'FixedMesh')")
    parser.add_argument("-f", help="Choose field to adapt to from {'s', 'f', 'b'}, denoting speed, free surface and both, resp.")
    parser.add_argument("-g", help="Apply metric gradation")
    parser.add_argument("-n", help="Specify number of mesh adaptations (default 1).")
    args = parser.parse_args()

    op = TurbineOptions()
    if args.a is not None:
        op.approach = args.a
    if args.f is not None:
        op.adapt_field = args.f
    if args.g is not None:
        op.gradate = bool(args.g)
    if args.n is not None:
        op.num_adapt = int(args.n)
    op.order_increase = True       # TODO: ExplicitErrorEstimator needs some work
    op.viscosity = 1.              # TODO: Increasing this value will smoothen and possibly help adjoint solver

    mesh2d = Mesh('channel.msh')
    if op.approach == "FixedMesh":
        solve_turbine(mesh2d, op=op)
    else:
        if op.approach == "HessianBased":
            for i in range(op.num_adapt):
                print("Generating solution on mesh {:d}".format(i))
                q = solve_turbine(mesh2d, op=op)
                mesh2d = mesh_adapt(q, op=op)
        else:
            for i in range(op.num_adapt):
                print("Generating solution on mesh {:d}".format(i))
                q, epsilon = get_error_estimators(mesh2d, op=op)
                with pyadjoint.stop_annotating():
                    mesh2d = mesh_adapt(q, epsilon, op=op)
        uv_2d, elev_2d = q.split()
        uv_2d, elev_2d = interp(mesh2d, uv_2d, elev_2d)
        File(op.directory() + 'AdaptedMeshSolution.pvd').write(uv_2d, elev_2d)
