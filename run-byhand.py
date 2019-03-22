from firedrake import *
import math
op2.init(log_level=INFO)

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

    def solve_onestep(self):
        g = self.g
        nu = self.nu
        n = self.n

        # function spaces etc
        u, eta = split(self.sol)
        z, zeta = TestFunctions(self.V)
        H = eta + self.b
        if norm(self.sol) < 1e-8:
            u_old = interpolate(as_vector((3., 0.)), self.V.sub(0))
        else:
            u_old = self.sol.split()[0].copy()

        F = 0
        # advection term
        F += inner(z, dot(u_old, nabla_grad(u)))*dx  # note use of u_old
        # pressure gradient term
        F += g*dot(z, grad(eta))*dx
        # viscosity term
        F += -nu*inner(z, dot(n, nabla_grad(u)))*ds
        F += nu*inner(grad(z), grad(u))*dx
        # drag term
        F += (self.C_D+self.C_b)*sqrt(dot(u_old, u_old))*dot(z, u)/H*dx  # note use of u_old
        # hudiv term (in.c Neumann condition)
        F += zeta*H*dot(u, n)*ds(1) + zeta*H*dot(u, n)*ds(2)
        F += -H*dot(u, grad(zeta))*dx

        # Dirichlet boundary conditions
        inflow = Constant((3., 0.))
        bc1 = DirichletBC(self.V.sub(0), inflow, 1)
        bc2 = DirichletBC(self.V.sub(1), 0, 2)

        # solve
        prob = NonlinearVariationalProblem(F, self.sol, bcs=[bc1, bc2])
        solv = NonlinearVariationalSolver(prob, solver_parameters=self.params)
        solv.solve()
        u, eta = self.sol.split()
        u.rename('Velocity')
        eta.rename('Elevation')

        # plot
        self.outfile.write(u, eta)

    def solve(self):
        print('Solving with assumed constant velocity')
        self.solve_onestep()
        print('Solving with previously established velocity')
        self.solve_onestep()

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
