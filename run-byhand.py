from firedrake import *
import math
#op2.init(log_level=INFO)

from adapt.adaptivity import *
from adapt.interpolation import interp
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
A_T = math.pi*(D/2)**2


class TurbineProblem():

    def __init__(self, mesh=None, op=TurbineOptions()):
        self.mesh = Mesh('channel.msh') if mesh is None else mesh
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
        self.n = FacetNormal(self.mesh)

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

        # plotting
        self.outfile = File('outputs/' + op.approach + '/sol.pvd')
        self.outfile_adjoint = File('outputs/' + op.approach + '/sol_adjoint.pvd')

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
        # hudiv term (in.c Neumann condition)
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

    def solve(self):
        print('Solving with assumed constant velocity')
        self.solve_onestep()
        print('Solving with previously established velocity')
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

    def objective_functional(self):
        u = self.sol.split()[0]
        unorm = sqrt(inner(u, u))
        self.J = assemble(self.C_D*(unorm**3)*dx)
        print("Objective functional: {:.4e}".format(self.J))
        return self.J


if __name__ == "__main__":

    tp = TurbineProblem()
    tp.solve()
    J = tp.objective_functional()
    tp.solve_adjoint()
