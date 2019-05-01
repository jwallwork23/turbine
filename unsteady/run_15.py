import math, sys
from subprocess import call
import os.path
from scipy.interpolate import interp1d
from numpy import loadtxt, savetxt


from thetis import *


#set_log_level(DEBUG)


outputDir = create_directory('output')
initial_meshname = '../fine_15_turbine.msh'
print_output('Loaded mesh ' + initial_meshname)
print_output('Exporting to ' + outputDir)

# total duration in seconds
T_tide = 1.24*3600
T_ramp = 1*T_tide
dt = 3
T = T_ramp+2*T_tide
# export interval in seconds
TExport = 30


# turbine parameters:
D = 20  # turbine diameter
C_T = 7.6  # thrust coefficient
# correction to account for the fact that the thrust coefficient is based on an upstream velocity
# whereas we are using a depth averaged at-the-turbine velocity (see Kramer and Piggott 2016, eq. (15))
H = 50 # max water depth
A_T = math.pi*(D/2)**2
correction = 4/(1+math.sqrt(1-A_T/(H*D)))**2
# NOTE, that we're not yet correcting power output here, so that will be overestimated 

mesh2d = Mesh(initial_meshname)
            

x = mesh2d.coordinates
# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
# bathymetry2d = Function(P1_2d, name="bathymetry")
bathymetry2d = Constant(50.)

bottom_drag = Constant(0.0025)
hViscosity = Constant(3)
#File("hvisco.pvd").write(hViscosity)


# --- create solver ---
solverObj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
options = solverObj.options
options.use_nonlinear_equations = True
options.simulation_export_time =  TExport
options.simulation_end_time = T
options.output_directory = outputDir
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.timestepper_type = 'CrankNicolson'
#options.timestepper_options.implicitness_theta = 1.0

options.use_lax_friedrichs_velocity = False
options.quadratic_drag_coefficient = bottom_drag
options.horizontal_viscosity = hViscosity
options.element_family = 'dg-cg'
options.timestep = dt 

# we've meshed the turbines as DxD squares, so we can treat it
# as turbine "farm"s with turbine density of 1 turbine per D^2 area
turbine_density = Constant(1.0/(D*5), domain=mesh2d)
farm_options = TidalTurbineFarmOptions()
farm_options.turbine_density = turbine_density
farm_options.turbine_options.diameter = D
farm_options.turbine_options.thrust_coefficient = C_T*correction
# assign ID 2..16 with the "farm"
for i in range(2, 17):
    options.tidal_turbine_farms[i] = farm_options

# callback that computes average power
#cb = turbines.TurbineFunctionalCallback(solverObj)
#solverObj.add_callback(cb, 'timestep')

# ---- Boundary conditions ------------
# boundary conditions
inflow_tag = 4
outflow_tag = 2
elev_func_in = Function(P1_2d)
elev_bc_in = {'elev': elev_func_in}
#solverObj.bnd_functions['shallow_water'] = {inflow_tag: elev_bc_in}
elev_func_out = Function(P1_2d)
elev_bc_out = {'elev': elev_func_out}
#solverObj.bnd_functions['shallow_water'] = {outflow_tag: elev_bc_out}
solverObj.bnd_functions['shallow_water'] = {inflow_tag: elev_bc_in, outflow_tag: elev_bc_out}

uv_init = as_vector((1.e-8, 0.0))
x = SpatialCoordinate(mesh2d)
elev_init = -1/3000 * x[0]

hmax = Constant(0.5)
omega = Constant(2*pi/T_tide)
tc = Constant(0)
def updateForcings(t):
  tc.assign(t)
  elev_func_in.assign(hmax*cos(omega*(tc-T_ramp)))
  elev_func_out.assign(hmax*cos(omega*(tc-T_ramp)+pi))
  print("DEBUG  t: %.0f" % t)
  
updateForcings(0.)
solverObj.assign_initial_conditions(uv=uv_init, elev=elev_init)
#solverObj.assign_initial_conditions(uv=uv_init)
#solverObj.timestepper.solver_parameters['snes_monitor'] = False
solverObj.iterate(update_forcings=updateForcings)
