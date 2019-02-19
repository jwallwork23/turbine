# simple set up with two turbines
from thetis_adjoint import *
from fenics_adjoint.solving import SolveBlock       # For extracting adjoint solutions
from fenics_adjoint.projection import ProjectBlock  # Exclude projections from tape reading
import pyadjoint
import math
# op2.init(log_level=INFO)

from adapt.adaptivity import *
from adapt.interpolation import interp
from turbine.options import TurbineOptions


# read global variables defining turbines from geo file
geo = open('channel.geo', 'r')
W = float(geo.readline().replace(';', '=').split('=')[1])
D = float(geo.readline().replace(';', '=').split('=')[1])
xt1 = float(geo.readline().replace(';', '=').split('=')[1])
xt2 = float(geo.readline().replace(';', '=').split('=')[1])
dt1 = float(geo.readline().replace(';', '=').split('=')[1])
dt2 = float(geo.readline().replace(';', '=').split('=')[1])
geo.close()
yt1=W/2
yt2=W/2


def solve_turbine(mesh2d, op=TurbineOptions()):
    """
    Solve steady state shallow water equations on mesh `mesh2d`, using `AdaptOptions` parameter
    class `op`.

    :return: approximate solution tuple for steady state shallow water equations.
    """
    # if we solve with PressureProjectionPicard (theta=1.0) it seems to converge (power output to 7
    # digits) in roughly 800 timesteps of 20s with SteadyState we only do 1 timestep (t_end should
    # be slightly smaller than timestep to achieve this)
    t_end = 0.9*op.dt

    H = 40  # water depth

    # turbine parameters:
    D = 18     # turbine diameter
    C_T = 0.8  # thrust coefficient

    # correction to account for the fact that the thrust coefficient is based on an upstream velocity
    # whereas we are using a depth averaged at-the-turbine velocity (see Kramer and Piggott 2016,
    # eq. (15))
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
    top_bottom_tag = 3
    freeslip_bc = {'un': Constant(0.)}
    noslip_bc = {'uv': Constant((0., 0.))}
    solver_obj.bnd_functions['shallow_water'] = {
        left_tag: {'uv': Constant((3., 0.))},
        # right_tag: {'un': Constant(3.), 'elev': Constant(0.)}
        right_tag: {'elev': Constant(0.)},
        # top_bottom_tag: freeslip_bc,
        # top_bottom_tag: noslip_bc,
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
    print("Average power: {p:.4e}".format(p=cb.average_power))

    return solver_obj


def get_error_estimators(mesh2d, op=TurbineOptions()):
    """
    Generate a posteriori error indicators on mesh `mesh2d` using `AdaptOptions` parameter class `op`.

    :return: approximate solution to steady state shallow water equations, a posteriori error estimate
    """
    # if we solve with PressureProjectionPicard (theta=1.0) it seems to converge (power output to 7
    # digits) in roughly 800 timesteps of 20s with SteadyState we only do 1 timestep (t_end should
    # be slightly smaller than timestep to achieve this)
    t_end = 0.9*op.dt

    H = 40. # water depth
    H_const = Constant(H)

    # Vector constants are broken in pyadjoint, so use a vector function instead
    P1 = VectorFunctionSpace(mesh2d, 'CG', 1)
    inflow = Function(P1).interpolate(as_vector([3., 0.]))
    noslip = Function(P1)

    # turbine parameters:
    D = 18     # turbine diameter
    C_T = 0.8  # thrust coefficient

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
    options.compute_residuals = op.approach == 'DWR'
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
    top_bottom_tag = 3
    freeslip_bc = {'un': Constant(0.)}
    noslip_bc = {'uv': noslip}
    solver_obj.bnd_functions['shallow_water'] = {
        # left_tag: {'uv': Constant((3., 0.))},
        left_tag: {'uv': inflow},
        # right_tag: {'un': Constant(3.), 'elev': Constant(0.)}
        right_tag: {'elev': Constant(0.)},
        # top_bottom_tag: freeslip_bc,
        # top_bottom_tag: noslip_bc,
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
    solver_obj.assign_initial_conditions(uv=as_vector((3.0, 0.0)))
    solver_obj.iterate()
    J = cb1.average_power
    print("Average power: {:.4e}".format(J))

    # Plot source term for adjoint equation
    if op.approach == 'AdjointOnly':
        u = solver_obj.fields.uv_2d
        P1 = VectorFunctionSpace(mesh2d, "CG", 1)
        unormu = project(u*sqrt(inner(u,u)), P1)
        File(op.directory() + 'AdjointSource2d.pvd').write(unormu)

    compute_gradient(J, Control(H_const))
    tape = get_working_tape()
    solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)
                                                    and not isinstance(block, ProjectBlock)
                                                    and block.adj_sol is not None]
    try:
        assert len(solve_blocks) == 1
    except:
        ValueError("Expected one SolveBlock, but encountered {:d}".format(len(solve_blocks)))

    with pyadjoint.stop_annotating():

        # Create function spaces and get primal and dual solutions
        V = op.mixed_space(mesh2d)
        adjoint = Function(V).assign(solve_blocks[0].adj_sol)
        adj_u, adj_eta = adjoint.split()
        adj_u.rename('adjoint_velocity_2d')
        adj_eta.rename('adjoint_elev_2d')
        File(op.directory() + "AdjointSolution2d.pvd").write(adj_u, adj_eta)

        if op.approach == 'AdjointOnly':
            return

        # Form error indicator
        P1 = FunctionSpace(mesh2d, "CG", 1)
        epsilon = Function(P1)
        if op.approach == "DWP":
            epsilon.project(inner(solver_obj.fields.solution_2d, adjoint))
        else:
            # TODO: Use z-z_h form
            ts = solver_obj.timestepper
            cell_res = ts.cell_residual(adjoint)
            print("Cell residual: {:.4e}".format(norm(cell_res)))
            edge_res = ts.edge_residual(adjoint)
            print("Edge residual: {:.4e}".format(norm(edge_res)))
            P0 = FunctionSpace(mesh2d, "DG", 0)
            cell_res = Function(P0).assign(cell_res)
            edge_res = Function(P0).assign(edge_res)
            # i = TestFunction(P0)
            # h = CellSize(mesh2d)

            # [Becker & Rannacher 2001] estimator
            #omega = assemble(i * (cell_res + edge_res) * dx)
            #rho = assemble(i * (cell_res * cell_res + (edge_res * edge_res) / sqrt(h)) * dx)
            #epsilon.project(omega*rho)

            # Adaptive strategies, as in [Rognes & Logg, 2010]
            if op.dwr_approach == 'error_representation':
                epsilon.project(cell_res + edge_res)
            elif op.dwr_approach == 'dwr':
                epsilon.project(cell_res + jump(edge_res))
            elif op.dwr_approach == 'cell_facet_split':
                epsilon.project(cell_res + abs(jump(edge_res)))
            else:
                raise ValueError("DWR approach {:s} not recognised.".format(op.dwr_approach))
            # TODO: Global higher-order approximation
            # TODO: Local higher-order approximation (patchwise interpolation). Use libsupermesh?
            # TODO: Difference quotient

            # [Ainsworth & Oden 1997] 'explicit' estimator
            # epsilon.project(assemble(i * (h * h * inner(cell_res, cell_res) + h * inner(edge_res, edge_res)) * dx))

            print("DWR estimator: {:.4e}".format(norm(epsilon)))
        epsilon = normalise_indicator(epsilon, op=op)
        epsilon.rename('error_2d')
        File(op.directory() + "ErrorIndicator2d.pvd").write(epsilon)
        tape.clear_tape()

    return solver_obj, epsilon


def mesh_adapt(solver_obj, error_indicator=None, metric=None, op=TurbineOptions()):
    """
    Adapt mesh based on an error indicator or field of interest.

    :arg solver_obj: Thetis FlowSolver2d object.
    :param error_indicator: optional error indicator upon which to adapt.
    :param metric: If a metric field is provided, it will be intersected.
    :param op: `AdaptOptions` parameter class.
    :return: adapted mesh and associated metric field.
    """
    mesh2d = solver_obj.mesh2d
    P1 = FunctionSpace(mesh2d, "CG", 1)

    if op.approach == 'HessianBased':
        uv_2d, elev_2d = solver_obj.fields.solution_2d.split()
        if op.adapt_field != 'elevation':       # metric for fluid speed
            spd = sqrt(inner(uv_2d, uv_2d))
            # spd = project(sqrt(inner(uv_2d, uv_2d)), P1)
            # spd = interpolate(sqrt(inner(uv_2d, uv_2d)), P1)
            M = steady_metric(spd, mesh=mesh2d, op=op)
        if op.adapt_field != 'fluid_speed':       # metric for free surface
            surf = elev_2d
            # surf = project(elev_2d, P1)
            # surf = interpolate(elev_2d, P1)
            M2 = steady_metric(surf, op=op)
        if op.adapt_field == 'both':       # intersect metrics for fluid speed and free surface
            M = metric_intersection(M, M2)
        elif op.adapt_field == 'elevation':
            M = M2

    elif op.approach in ('DWP', 'DWR'):
        assert(error_indicator is not None)
        M = isotropic_metric(error_indicator, op=op)

    if op.intersect_boundary:
        bdy = 'on_boundary'  # use boundary tags to gradate to individual boundaries
        H0 = project(CellSize(mesh2d), P1)
        M_ = isotropic_metric(H0, bdy=bdy, op=op)  # Initial boundary metric
        M = metric_intersection(M, M_, bdy=bdy)
    if op.gradate:
        M = gradate_metric(M, op=op)

    if metric is not None:
        M = metric_intersection(M, metric)

    mesh2d = AnisotropicAdaptation(mesh2d, M).adapted_mesh
    print("Number of elements after mesh adaptation: {:d}".format(mesh2d.num_cells()))

    return mesh2d, M


if __name__ == "__main__":

    import argparse
    from time import clock
    import datetime

    now = datetime.datetime.now()
    date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

    parser = argparse.ArgumentParser()
    parser.add_argument("-approach", help="Choose adaptive approach from {'HessianBased', 'DWP', 'DWR'} (default 'FixedMesh')")
    parser.add_argument("-field", help="Choose field to adapt to from {'fluid_speed', 'elevation', 'both'}, denoting speed, free surface and both, resp.")
    parser.add_argument("-gradate", help="Apply metric gradation")
    parser.add_argument("-intersect", help="Intersect with previous metric")
    parser.add_argument("-intersect_boundary", help="Intersect with initial boundary metric")
    parser.add_argument("-n", help="Specify number of mesh adaptations (default 1).")
    parser.add_argument("-m", help="Toggle additional message for output file")
    parser.add_argument("-dwr_approach", help="DWR error estimation approach")
    args = parser.parse_args()

    op = TurbineOptions()
    if args.approach is not None:
        op.approach = args.approach
    if args.field is not None:
        op.adapt_field = args.field
    if args.gradate is not None:
        op.gradate = bool(args.gradate)
    if args.intersect is not None:
        op.intersect = bool(args.intersect)
    if args.intersect_boundary is not None:
        op.intersect_boundary = bool(args.intersect_boundary)
    if args.drag_coefficient is not None:
        op.drag_coefficient = float(args.drag_coefficient)
    if args.viscosity is not None:
        op.viscosity = float(args.viscosity)
    if args.bc is not None:
        op.north_south_bc = args.bc
    if args.n is not None:
        op.num_adapt = int(args.n)
    if args.dwr_approach is not None:
        op.dwr_approach = args.dwr_approach

    mesh2d = Mesh('channel.msh')
    op.target_vertices = mesh2d.num_vertices() * op.rescaling  # NOTE: This is not done each step

    if op.approach != 'AdjointOnly':
        f = open(op.directory() + 'data/' + date + '.txt', 'a')
        if args.m is not None:
            label = input('Description for this run: ')
            f.write(label+'\n\n')
        f.write('MESH 0\n')
        f.write('outer resolution: {:.3f}\n'.format(dt1))
        f.write('inner resolution: {:.3f}\n'.format(dt2))
        f.write('mesh elements:    {:d}\n'.format(mesh2d.num_cells()))
        f.write('mesh edges:       {:d}\n'.format(mesh2d.num_edges()))
        f.write('mesh vertices:    {:d}\n\n'.format(mesh2d.num_vertices()))

    M = None
    tic = clock()
    if op.approach == "FixedMesh":
        with pyadjoint.stop_annotating():
            solve_turbine(mesh2d, op=op)
    elif op.approach == 'AdjointOnly':
        get_error_estimators(mesh2d, op=op)
    else:
        File(op.directory() + 'Mesh0.pvd').write(mesh2d.coordinates)
        if op.approach == "HessianBased":
            with pyadjoint.stop_annotating():
                for i in range(op.num_adapt):
                    print("Generating solution on mesh {:d}".format(i))
                    solve_time = clock()
                    solver_obj = solve_turbine(mesh2d, op=op)
                    solve_time = clock() - solve_time
                    adapt_time = clock()
                    if M is not None:
                        M = interp(mesh2d, M)
                    mesh2d, M = mesh_adapt(solver_obj, metric=M, op=op)
                    adapt_time = clock() - adapt_time
                    f.write('MESH {:d}\n'.format(i+1))
                    f.write('mesh elements: {:d}\n'.format(mesh2d.num_cells()))
                    f.write('mesh edges:    {:d}\n'.format(mesh2d.num_edges()))
                    f.write('mesh vertices: {:d}\n'.format(mesh2d.num_vertices()))
                    f.write('solver time:   {:.3f}\n'.format(solve_time))
                    f.write('adapt time:    {:.3f}\n\n'.format(adapt_time))
                    File(op.directory() + 'Mesh' + str(i+1) + '.pvd').write(mesh2d.coordinates)
                    if not op.intersect:
                        M = None
        else:
            eps_norm_ = 1.
            eps_norm = 0.
            # while abs(eps_norm_ / eps_norm) - 1. > 1e-4:  TODO: This or #elements converges
            for i in range(op.num_adapt):
                eps_norm_ = eps_norm
                print("Generating solution on mesh {:d}".format(i))
                solve_time = clock()
                solver_obj, epsilon = get_error_estimators(mesh2d, op=op)
                solve_time = clock() - solve_time
                adapt_time = clock()
                if M is not None:
                    M = interp(mesh2d, M)
                with pyadjoint.stop_annotating():
                    mesh2d, M = mesh_adapt(solver_obj, error_indicator=epsilon, metric=M, op=op)
                adapt_time = clock() - adapt_time
                f.write('MESH {:d}\n'.format(i+1))
                f.write('mesh elements: {:d}\n'.format(mesh2d.num_cells()))
                f.write('mesh edges:    {:d}\n'.format(mesh2d.num_edges()))
                f.write('mesh vertices: {:d}\n'.format(mesh2d.num_vertices()))
                f.write('solver time:   {:.3f}\n'.format(solve_time))
                f.write('adapt time:    {:.3f}\n'.format(adapt_time))
                eps_norm = norm(epsilon)
                f.write('indicator:     {:.4e}\n\n'.format(eps_norm))
                if op.approach == 'DWP':
                    meshfile = op.directory() + 'Mesh' + str(i+1) + '.pvd'
                else:
                    meshfile = op.directory() + op.dwr_approach + '_mesh' + str(i+1) + '.pvd'
                File(meshfile).write(mesh2d.coordinates)
                if not op.intersect:
                    M = None
    if op.approach != 'AdjointOnly':
        toc = clock()
        f.write('SUMMARY\n')
        f.write('total time:   {:.3f}\n'.format(toc-tic))
        f.write('viscosity:    {:.3f}\n'.format(op.viscosity))
        f.write('drag coeff.:  {:.3f}\n'.format(op.drag_coefficient))
        f.write('N/S bcs:      {:s}\n'.format(op.north_south_bc))
        if op.approach != 'FixedMesh':
            f.write('mesh adapts:  {:d}\n'.format(op.num_adapt))
            f.write('gradation:    {}\n'.format(op.gradate))
            f.write('intersect:    {}\n'.format(op.intersect))
            if op.approach == 'HessianBased':
                f.write('adapt field:  {:s}\n'.format(op.adapt_field))
            elif op.approach == 'DWR':
                f.write('dwr approach: {:s}\n'.format(op.dwr_approach))
        f.write('\n\n')
        f.close()
