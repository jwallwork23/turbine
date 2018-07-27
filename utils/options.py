from thetis import *
from thetis.configuration import *

import numpy as np

__all__ = ["TohokuOptions", "RossbyWaveOptions", "KelvinWaveOptions", "GaussianOptions", "AdvectionOptions"]


class AdaptOptions(FrozenConfigurable):
    name = 'Common parameters for TohokuAdapt project'

    # Mesh adaptivity parameters
    approach = Unicode('FixedMesh',
                       help="Mesh adaptive approach considered, from {'FixedMesh', 'HessianBased', 'DWP', 'DWR'}"
                       ).tag(config=True)
    gradate = Bool(False, help='Toggle metric gradation.').tag(config=True)
    adapt_on_bathymetry = Bool(False, help='Toggle adaptation based on bathymetry field.').tag(config=True)
    plot_pvd = Bool(False, help='Toggle plotting of fields.').tag(config=True)
    plot_metric = Bool(False, help='Toggle plotting of metric field.').tag(config=True)
    max_element_growth = PositiveFloat(1.4, help="Metric gradation scaling parameter.").tag(config=True)
    max_anisotropy = PositiveFloat(100., help="Maximum tolerated anisotropy.").tag(config=True)
    num_adapt = NonNegativeInteger(1, help="Number of mesh adaptations per remeshing.").tag(config=True)
    order_increase = Bool(False, help="Interpolate adjoint solution into higher order space.").tag(config=True)
    normalisation = Unicode('lp', help="Normalisation approach, from {'lp', 'manual'}.").tag(config=True)
    hessian_recovery = Unicode('dL2', help="Hessian recovery technique, from {'dL2', 'parts'}.").tag(config=True)
    timestepper = Unicode('CrankNicolson', help="Time integration scheme used.").tag(config=True)
    norm_order = NonNegativeInteger(2, help="Degree p of Lp norm used.")
    family = Unicode('dg-dg', help="Mixed finite element family, from {'dg-dg', 'dg-cg'}.").tag(config=True)
    min_norm = PositiveFloat(1e-6).tag(config=True)
    max_norm = PositiveFloat(1e9).tag(config=True)

    def final_index(self):
        return int(np.ceil(self.end_time / self.timestep))  # Final timestep index

    def first_export(self):
        return int(self.start_time / (self.timesteps_per_export * self.timestep))  # First exported timestep of period of interest

    def final_export(self):
        return int(self.final_index() / self.timesteps_per_export)  # Final exported timestep of period of interest

    def final_mesh_index(self):
        return int(self.final_index() / self.timesteps_per_remesh)  # Final mesh index

    def exports_per_remesh(self):
        assert self.timesteps_per_remesh % self.timesteps_per_export == 0
        return int(self.timesteps_per_remesh / self.timesteps_per_export)

    def mixed_space(self, mesh, enrich=False):
        """
        :param mesh: mesh upon which to build mixed space.
        :return: mixed VectorFunctionSpace x FunctionSpace as specified by ``self.family``.
        """
        d1 = 1
        d2 = 2 if self.family == 'dg-cg' else 1
        if enrich:
            d1 += self.order_increase
            d2 += self.order_increase
        return VectorFunctionSpace(mesh, "DG", d1) * FunctionSpace(mesh, "DG" if self.family == 'dg-dg' else "CG", d2)

    def adaptation_stats(self, mn, adaptTimer, solverTime, nEle, Sn, mM, t):
        """
        :arg mn: mesh number.
        :arg adaptTimer: time taken for mesh adaption.
        :arg solverTime: time taken for solver.
        :arg nEle: current number of elements.
        :arg Sn: sum over #Elements.
        :arg mM: tuple of min and max #Elements.
        :arg t: current simuation time.
        :return: mean element count.
        """
        av = Sn / mn
        print("""\n************************** Adaption step %d ****************************
Percent complete  : %4.1f%%    Adapt time : %4.2fs Solver time : %4.2fs     
#Elements... Current : %d  Mean : %d  Minimum : %s  Maximum : %s\n""" %
              (mn, 100 * t / self.end_time, adaptTimer, solverTime, nEle, av, mM[0], mM[1]))
        return av

    def directory(self):
        return 'outputs/' + self.approach + '/'


class TurbineOptions(AdaptOptions):
    name = 'Parameters for the 2 turbine problem'
    mode = 'Turbine'

    # Solver parameters
    dt = PositiveFloat(20.).tag(config=True)
    target_vertices = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    adapt_field = Unicode('s', help="Adaptation field of interest, from {'s' (speed), 'f' (free surface), 'b' (both)}.").tag(config=True)
    h_min = PositiveFloat(1e-4, help="Minimum element size").tag(config=True)
    h_max = PositiveFloat(10., help="Maximum element size").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    viscosity = NonNegativeFloat(0.1).tag(config=True)
    drag_coefficient = NonNegativeFloat(0.0025).tag(config=True)
