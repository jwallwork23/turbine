from firedrake import *
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
dx1 = float(geo.readline().replace(';', '=').split('=')[1])
dx2 = float(geo.readline().replace(';', '=').split('=')[1])
L = float(geo.readline().replace(';', '=').split('=')[1])
geo.close()
yt1=W/2
yt2=W/2

def solve_turbine(mesh2d, op=TurbineOptions()):

    # physical parameters
    b = Constant(40.)
    nu = Constant(op.viscosity)
    C_b = Constant(op.drag_coefficient)
    g = Constant(9.81)

    # turbine parameters
    D = 18
    A_T = math.pi*(D/2)**2
    correction = 4/(1+math.sqrt(1-A_T/(b*D)))**2
    C_T = op.thrust_coefficient * correction
    op.region_of_interest = [(xt1, yt1, D/2), (xt2, yt2, D/2)]
    turbine_density = op.bump(mesh2d, scale=2./assemble(op.bump(mesh2d)*dx))
    C_D = Constant(C_T * A_T / 2.) * turbine_density

    # function spaces etc
    Taylor_Hood = VectorFunctionSpace(mesh2d, "CG", 2) * FunctionSpace(mesh2d, "CG", 1)
    q = Function(Taylor_Hood)
    u, eta = split(q)
    z, zeta = TestFunctions(Taylor_Hood)
    n = FacetNormal(mesh2d)
    H = eta + b
    u_old = interpolate(as_vector((3., 0.)), Taylor_Hood.sub(0))

    F = 0
    # advection term
    F += inner(z, dot(u_old, nabla_grad(u)))*dx  # note use of u_old
    # pressure gradient term
    F += g*dot(z, grad(eta))*dx
    # viscosity term
    F += -nu*inner(z, dot(n, nabla_grad(u)))*ds
    F += nu*inner(grad(z), grad(u))*dx
    # drag term
    F += (C_D+C_b)*sqrt(dot(u_old, u_old))*dot(z, u)/H*dx  # note use of u_old
    # hudiv term (in.c Neumann condition)
    F += zeta*H*dot(u, n)*ds(1) + zeta*H*dot(u, n)*ds(2)
    F += -H*dot(u, grad(zeta))*dx

    # Dirichlet boundary conditions
    inflow = Constant((3., 0.))
    bc1 = DirichletBC(Taylor_Hood.sub(0), inflow, 1)
    bc2 = DirichletBC(Taylor_Hood.sub(1), 0, 2)

    # solve
    params = {
              'snes_type': 'newtonls',
              #'ksp_type': 'gmres',
              'ksp_type': 'preonly',
              'pc_type': 'lu',
              #'pc_factor_mat_solver_type': 'mumps',
              'mat_type': 'aij',
              'snes_monitor': None,
              'ksp_monitor': None,
              'ksp_converged_reason': None,
             }
    prob = NonlinearVariationalProblem(F, q, bcs=[bc1, bc2])
    solv = NonlinearVariationalSolver(prob, solver_parameters=params)
    solv.solve()
    u, eta = q.split()
    u.rename('Velocity')
    eta.rename('Elevation')

    # plot
    File('outputs/' + op.approach + '/sol.pvd').write(u, eta)

    # objective functional
    unorm = sqrt(inner(u, u))
    J = assemble(C_D*(unorm**3)*dx)

    return J


if __name__ == "__main__":
    mesh = Mesh('channel.msh')
    solve_turbine(mesh)
