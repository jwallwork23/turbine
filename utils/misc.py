from firedrake import *

from .options import TurbineOptions


__all__ = ["index_string", "subdomain_indicator"]


def index_string(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def subdomain_indicator(mesh, op=TurbineOptions()):
    n = len(op.turbine_geometries)

    actuator_domains = []
    sub_dx = 1.0 / n
    sub_dy = sub_dx
    for i in range(0, int(n)):
        for j in range(0, int(n)):
            sub_xa = i * sub_dx
            sub_xb = (i + 1) * sub_dx
            sub_ya = j * sub_dy
            sub_yb = (j + 1) * sub_dy
            actuator_domains.append(
                SubDomainData(
                    reduce(ufl.And,
                           [x[0] >= sub_xa,
                            x[0] <= sub_xb,
                            x[1] >= sub_ya,
                            x[1] <= sub_yb])
                )
            )

    marker = Function(FunctionSpace(mesh, "DG", 0))
    for i, actuator in enumerate(actuator_domains):
        marker.interpolate(Constant(i + 1), subset=actuator)
        File(op.directory() + 'subdomains.pvd').write(marker)