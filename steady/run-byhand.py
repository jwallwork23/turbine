from firedrake import *
import math
#op2.init(log_level=INFO)

from adapt_utils.adapt.metric import *
from adapt_utils.adapt.interpolation import *
from adapt_utils.turbine.options import TurbineOptions


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
A_T = math.pi*(D/2)**2


class TurbineProblem():

    def __init__(self, mesh=None, op=TurbineOptions()):
        self.mesh = Mesh('channel.msh') if mesh is None else mesh
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.h = CellSize(self.mesh)
        self.n = FacetNormal(self.mesh)
        self.op = op

        # physical parameters
        self.b = Constant(40.)
        self.nu = Constant(op.viscosity)
        self.C_b = Constant(op.drag_coefficient)
        self.g = Constant(9.81)

        # turbine parameters
        correction = 4/(1+math.sqrt(1-A_T/(self.b*D)))**2
        C_T = op.thrust_coefficient * correction
        self.op.region_of_interest = [(xt1, yt1, D/2), (xt2, yt2, D/2)]
        turbine_density = self.op.bump(self.mesh, scale=2./assemble(self.op.bump(self.mesh)*dx))
        self.C_D = Constant(C_T * A_T / 2.) * turbine_density

        # Taylor-Hood space
        self.V = VectorFunctionSpace(self.mesh, "CG", 2) * FunctionSpace(self.mesh, "CG", 1)
        self.sol = Function(self.V)
        u, eta = self.sol.split()
        u.rename('Velocity')
        eta.rename('Elevation')
        self.sol_adjoint = Function(self.V)
        z, zeta = self.sol_adjoint.split()
        z.rename('Adjoint velocity')
        zeta.rename('Adjoint elevation')

        # solver parameters
        self.params = {
                       'snes_type': 'newtonls',
                       #'ksp_type': 'gmres',
                       'ksp_type': 'preonly',
                       'pc_type': 'lu',
                       #'pc_factor_mat_solver_type': 'mumps',
                       'mat_type': 'aij',
                       'snes_monitor': None,
                       'ksp_monitor': None,
                       #'ksp_converged_reason': None,
                      }

        # plotting  # FIXME: consistency
        self.di = self.op.directory()
        self.outfile = File(self.di + 'sol.pvd')
        self.outfile_adjoint = File(self.di + 'sol_adjoint.pvd')
        self.outfile_indicator = File(self.di + 'indicator.pvd')

    def solve_onestep(self):
        g = self.g
        nu = self.nu
        n = self.n
        u, eta = split(self.sol)
        psi, phi = TestFunctions(self.V)
        H = eta + self.b
        if norm(self.sol) < 1e-8:
            u_old = interpolate(as_vector((3., 0.)), self.V.sub(0))
        else:
            u_old = self.sol.split()[0].copy()

        F = 0
        # advection term
        F += inner(psi, dot(u_old, nabla_grad(u)))*dx  # note use of u_old
        # pressure gradient term
        F += g*dot(psi, grad(eta))*dx
        # viscosity term
        F += -nu*inner(psi, dot(n, nabla_grad(u)))*ds
        F += nu*inner(grad(psi), grad(u))*dx
        # drag term
        F += (self.C_D+self.C_b)*sqrt(dot(u_old, u_old))*dot(psi, u)/H*dx  # note use of u_old
        # hudiv term (inc. Neumann condition)
        F += phi*H*dot(u, n)*ds(1) + phi*H*dot(u, n)*ds(2)
        F += -H*dot(u, grad(phi))*dx

        # Dirichlet boundary conditions
        inflow = Constant((3., 0.))
        bc1 = DirichletBC(self.V.sub(0), inflow, 1)
        bc2 = DirichletBC(self.V.sub(1), 0, 2)

        # solve
        prob = NonlinearVariationalProblem(F, self.sol, bcs=[bc1, bc2])
        solv = NonlinearVariationalSolver(prob, solver_parameters=self.params)
        solv.solve()

        # plot
        u, eta = self.sol.split()
        self.outfile.write(u, eta)

    def solve(self, prev_sol=None):
        if prev_sol is None:
            print('Solving with assumed constant velocity')
            self.solve_onestep()
            print('Solving with previously established velocity')
        else:
            self.sol = prev_sol
            print('Solving with provided velocity')
        self.solve_onestep()

    def solve_adjoint(self):
        g = self.g
        nu = self.nu
        n = self.n
        u, eta = split(self.sol)
        z, zeta = TrialFunctions(self.V)
        psi, phi = TestFunctions(self.V)
        H = eta + self.b
        C = self.C_D + self.C_b
        unorm = sqrt(inner(u, u))

        a = 0
        # LHS contributions from adjoint momentum equation
        a += inner(dot(transpose(grad(u)), z), psi)*dx
        a += -inner(div(u)*z, psi)*dx
        a += -inner(dot(u, nabla_grad(z)), psi)*dx
        #a += inner(z, grad(dot(u, psi)))*dx
        #a += -inner(div(nu*grad(z)), psi)*dx
        a += nu*inner(grad(z), grad(psi))*dx
        a += -inner(H*grad(zeta), psi)*dx
        #a += zeta*div(H*psi)*dx
        a += C*(unorm*inner(z, psi) + inner(u, z)*inner(u, psi)/unorm)/H*dx

        # LHS contributions from adjoint continuity equation
        a += g*div(z)*phi*dx
        #a += -g*inner(z, grad(phi))*dx
        #a += inner(u, grad(zeta))*phi*dx
        a += -zeta*div(phi*u)*dx
        a += C*unorm*inner(u, z)*phi/(H*H)*dx  # TODO: sign?

        # RHS
        L = 3.*self.C_D*unorm*inner(u, psi)*dx

        # boundary conditions  # FIXME: currently using heuristics
        #a += -inner(u*dot(z, n), psi)*ds(1)
        #a += -inner(u*dot(z, n), psi)*ds(3)
        #a += -dot(n*zeta*H, psi)*ds(1)
        #a += -dot(n*zeta*H, psi)*ds(3)
        a += -inner(nu*dot(n, nabla_grad(z)), psi)*ds(1)
        a += -inner(nu*dot(n, nabla_grad(z)), psi)*ds(2)
        a += -inner(nu*dot(n, nabla_grad(z)), psi)*ds(3)
        #a += -nu*inner(psi, dot(n, nabla_grad(z)))*ds(1)
        #a += -nu*inner(psi, dot(n, nabla_grad(z)))*ds(3)
        #a += -g*phi*dot(z, n)*ds(2)
        ##a += -zeta*phi*dot(u, n)*ds(1)
        #a += -zeta*phi*dot(u, n)*ds(2)

        a += -phi*zeta*dot(u, n)*ds(1) -phi*zeta*dot(u, n)*ds(2)

        # solve
        params = {
                  'mat_type': 'aij',
                  'pc_type': 'lu',
                  'ksp_monitor': None,
                  #'ksp_converged_reason': None,
                 }
        #bc = DirichletBC(self.V.sub(0), 0, 'on_boundary')  # FIXME: not apparent in discrete adjoint
        bc = DirichletBC(self.V.sub(0), interpolate(u, self.V.sub(0)), 2)  # FIXME: heuristic
        solve(a == L, self.sol_adjoint, bcs=bc, solver_parameters=params)

        # plot
        z, zeta = self.sol_adjoint.split()
        self.outfile_adjoint.write(z, zeta)

    def objective_functional(self, sol=None):
        u = self.sol.split()[0] if sol is None else sol.split()[0]
        unorm = sqrt(inner(u, u))
        self.J = assemble(self.C_D*(unorm**3)*dx)
        msg = 'O' if sol is None else 'Interpolated o'
        print("{:s}bjective functional: {:.4e}".format(msg, self.J))
        return self.J

    def get_hessian_metric(self, adjoint=False):
        u, eta = self.sol_adjoint.split() if adjoint else self.sol.split()
        if self.op.adapt_field in ('fluid_speed', 'both'):
            spd = interpolate(sqrt(inner(u, u)), self.P1)
            self.M = steady_metric(spd, op=self.op)
        if self.op.adapt_field == 'fluid_speed_cubed':
            spd = interpolate(sqrt(inner(u, u))**3, self.P1)
            self.M = steady_metric(spd, op=self.op)
        if self.op.adapt_field == 'elevation':
            self.M = steady_metric(eta, op=self.op)
        elif self.op.adapt_field == 'both':
            self.M = metric_intersection(self.M, steady_metric(eta, op=self.op))

    def get_hessian_metric_superposed(self):
        self.get_hessian_metric(adjoint=False)
        M = self.M.copy()
        self.get_hessian_metric(adjoint=True)
        self.M = metric_intersection(M, self.M)

    def vorticity_indication(self):
        u = self.sol.split()[0]
        self.indicator = project(curl(u), self.P1)

    def explicit_estimation(self):
        g = self.g
        nu = self.nu
        n = self.n
        i = TestFunction(self.P0)
        u, eta = self.sol.split()
        unorm = sqrt(inner(u, u))
        H = eta + self.b
        C = self.C_D + self.C_b

        # strong residuals (on cells)
        R0 = div(nu*grad(u)) - dot(u, nabla_grad(u)) - g*grad(eta) - C*unorm*u/H
        R1 = -div(H*u)
        R_norm = assemble(i*(inner(R0, R0) + R1*R1)*dx)

        # solve auxiliary problem to assemble edge residual
        r_norm = TrialFunction(self.P0)
        mass_term = i*r_norm*dx
        r = -H*dot(u, n)         # Neumann condition on banks
        flux_terms = ((r*r*i)('+') + (r*r*i)('-'))*dS + r*r*i*ds(3)
        r = Constant(3.) - u[0]  # Dirichlet condition on inflow
        flux_terms += i*r*r*ds(1)
        r = -eta                 # Dirichlet condition on outflow
        flux_terms += i*r*r*ds(2)
        r_norm = Function(self.P0)
        solve(mass_term == flux_terms, r_norm)

        # form error estimator
        self.indicator = project(sqrt(self.h*self.h*R_norm + 0.5*self.h*r_norm), self.P0)
        self.indicator.rename('explicit')

    def explicit_estimation_adjoint(self):
        g = self.g
        nu = self.nu
        i = TestFunction(self.P0)

        u, eta = self.sol.split()
        z, zeta = self.sol_adjoint.split()
        unorm = sqrt(inner(u, u))
        H = eta + self.b
        C = self.C_D + self.C_b

        # strong residuals (on cells)
        R0 = 3*self.C_D*unorm*u - dot(transpose(grad(u)), z) + div(u)*z + dot(u, nabla_grad(z)) + div(nu*grad(z)) + H*grad(zeta) - C*(unorm*z + inner(u, z)*u/unorm)/H
        R1 = g*div(z) + inner(u, grad(zeta)) + C*unorm*inner(u, z)/H
        R_norm = assemble(i*(inner(R0, R0) + R1*R1)*dx)

        # TODO: flux terms
        r_norm = Constant(0.)

        # form error estimator
        self.indicator = project(sqrt(self.h*self.h*R_norm + 0.5*self.h*r_norm), self.P0)
        self.indicator.rename('explicit')

    def dwp_indication(self):
        self.indicator = project(inner(self.sol, self.sol_adjoint), self.P1)
        self.indicator.rename('dwp')

    def difference_quotient_estimation(self):
        raise NotImplementedError  # TODO
 
    def dwr_estimation(self):
        raise NotImplementedError  # TODO

    def dwr_estimation_adjoint(self):
        raise NotImplementedError  # TODO

    def ensure_p1(self):
        el = self.indicator.ufl_element()
        if (el.family(), el.degree()) != ('Lagrange', 1):
            self.indicator = project(self.indicator, self.P1)

    def normalise_indicator(self):
        name = self.indicator.name()
        self.ensure_p1()
        self.indicator = normalise_indicator(self.indicator, op=self.op)
        self.indicator.rename(name + '_indicator')
        self.outfile_indicator.write(self.indicator)

    def get_isotropic_metric(self):
        self.normalise_indicator()
        self.M = isotropic_metric(self.indicator, op=self.op)

    def get_anisotropic_metric(self, adjoint=False):
        self.ensure_p1()

        # construct Hessian
        u, eta = self.sol_adjoint.split() if adjoint else self.sol.split()
        if self.op.adapt_field in ('fluid_speed', 'both'):
            spd = interpolate(sqrt(inner(u, u)), self.P1)
            self.M = construct_hessian(spd, op=self.op)
        if self.op.adapt_field == 'fluid_speed_cubed':
            spd = interpolate(sqrt(inner(u, u))**3, self.P1)
            self.M = steady_metric(spd, op=self.op)
        if self.op.adapt_field == 'elevation':
            self.M = construct_hessian(eta, op=self.op)
        elif self.op.adapt_field == 'both':
            raise NotImplementedError  # TODO

        # scale with error indicator
        for i in range(len(self.indicator.dat.data)):   # TODO: use pyop2
            self.M.dat.data[i][:,:] *= self.indicator.dat.data[i]
        self.M = steady_metric(None, H=self.M, op=self.op)

    def adapt_mesh(self, relaxation_parameter=0.9, prev_metric=None, plot_mesh=True):
        
        # Estimate error and generate associated metric
        if self.op.approach == 'hessian':
            self.get_hessian_metric(adjoint=False)
        elif self.op.approach == 'hessian_adjoint':
            self.get_hessian_metric(adjoint=True)
        elif self.op.approach == 'hessian_superposed':
            self.get_hessian_metric_superposed()
        elif self.op.approach == 'vorticity':
            self.vorticity_indication()
            self.get_isotropic_metric()
        elif self.op.approach == 'explicit':
            self.explicit_estimation()
            self.get_isotropic_metric()
        elif self.op.approach == 'explicit_adjoint':
            self.explicit_estimation_adjoint()
            self.get_isotropic_metric()
        elif self.op.approach == 'explicit_superposed':
            self.explicit_estimation()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.explicit_estimation_adjoint()
            self.get_isotropic_metric()
            self.M = metric_intersection(M, self.M)
        elif self.op.approach == 'explicit_hessian':
            self.explicit_estimation()
            self.get_anisotropic_metric(adjoint=True)
        elif self.op.approach == 'explicit_hessian_adjoint':
            self.explicit_estimation_adjoint()
            self.get_anisotropic_metric(adjoint=False)
        elif self.op.approach == 'explicit_hessian_superposed':
            self.explicit_estimation()
            self.get_anisotropic_metric(adjoint=True)
            M = self.M.copy()
            self.explicit_estimation_adjoint()
            self.get_anisotropic_metric(adjoint=False)
            self.M = metric_intersection(M, self.M)
        elif self.op.approach == 'dwp':
            self.dwp_indication()
            self.get_isotropic_metric()
        elif self.op.approach == 'dwr':
            self.dwr_estimation()
            self.get_isotropic_metric()
        elif self.op.approach == 'dwr_adjoint':
            self.dwr_estimation_adjoint()
            self.get_isotropic_metric()
        elif self.op.approach == 'dwr_both':
            self.dwr_estimation()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_estimation_adjoint()
            self.indicator.interpolate(Constant(0.5)*(i+self.indicator))
            self.get_isotropic_metric()
        elif self.op.approach == 'dwr_averaged':
            self.dwr_estimation()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_estimation_adjoint()
            self.indicator.interpolate(Constant(0.5)*(abs(i)+abs(self.indicator)))
            self.get_isotropic_metric()
        elif self.op.approach == 'dwr_relaxed':
            self.dwr_estimation()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_estimation_adjoint()
            self.get_isotropic_metric()
            self.M = metric_relaxation(M, self.M)
        elif self.op.approach == 'dwr_superposed':
            self.dwr_estimation()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_estimation_adjoint()
            self.get_isotropic_metric()
            self.M = metric_intersection(M, self.M)
        else:
            raise ValueError("Adaptivity mode {:s} not regcognised.".format(self.op.approach))

        # Apply metric relaxation, if requested
        self.M_unrelaxed = self.M.copy()
        if prev_metric is not None:
            self.M.project(metric_relaxation(interp(self.mesh, prev_metric), self.M, relaxation_parameter))
        # (Default relaxation of 0.9 following [Power et al 2006])
            
        # Adapt mesh
        self.mesh = AnisotropicAdaptation(self.mesh, self.M).adapted_mesh
        if plot_mesh:
            File(self.di + 'Mesh.pvd').write(self.mesh.coordinates)     


if __name__ == "__main__":

    import argparse
    import datetime

    now = datetime.datetime.now()
    date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

    parser = argparse.ArgumentParser()
    parser.add_argument("-approach", help="Choose adaptive approach from {'HessianBased', 'DWP', 'DWR'} (default 'FixedMesh'). Option 'AdjointOnly' allows to just look at adjoint solution.")
    parser.add_argument("-field", help="Choose field to adapt to from {'fluid_speed', 'elevation', 'both'}, denoting speed, free surface and both, resp.")
    parser.add_argument("-n", help="Specify number of mesh adaptations (default 1).")
    parser.add_argument("-m", help="Message for output file")
    args = parser.parse_args()

    op = TurbineOptions()
    if args.approach is not None:
        op.approach = args.approach
    if args.field is not None:
        op.adapt_field = args.field
    op.num_adapt = 1
    if args.n is not None:
        op.num_adapt = int(args.n)

    # TODO: stopping criteria
    prev_sol = None  
    mesh = RectangleMesh(100, 20, L, W) if op.approach == 'fixed_mesh' else Mesh('channel.msh')
    tp = TurbineProblem(mesh=mesh, op=op)
    logfile = open(tp.di + 'log', 'a+')
    logfile.write(date + '{:s}\n\n'.format(' ' + args.m if args.m is not None else ''))
    logfile.write('Mesh  0: elements = {:10d}\n'.format(tp.mesh.num_cells()))
    for i in range(op.num_adapt):
        print('Solving on mesh {:d}'.format(i))
        tp.solve(prev_sol=prev_sol)
        J = tp.objective_functional()
        logfile.write('Mesh {:2d}:        J = {:.4e}\n'.format(i, J))
        if op.approach != 'fixed_mesh':
            tp.solve_adjoint()
            tp.adapt_mesh()
        else:
            tp.mesh = RectangleMesh((i+2)*100, (i+2)*20, L, W)
        prev_sol = tp.sol
        tp = TurbineProblem(mesh=tp.mesh, op=op)
        logfile.write('Mesh {:2d}: elements = {:10d}\n'.format(i+1, tp.mesh.num_cells()))
        prev_sol = mixed_pair_interp(tp.V, prev_sol)
        J = tp.objective_functional(sol=prev_sol)
        logfile.write('Mesh {:2d}: J_interp = {:.4e}\n'.format(i+1, J))
    logfile.write('\n\n')
    logfile.close()

    # print logfile to screen
    logfile = open(tp.di + 'log', 'r')
    for line in logfile:
        print(line.split('\n')[0])
    logfile.close()
