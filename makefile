all: mesh

mesh:
	gmsh -2 coarse_1_turbine.geo
	gmsh -2 coarse_2_turbine.geo
	gmsh -2 coarse_2_turbine_centred.geo
	gmsh -2 fine_2_turbine.geo
	gmsh -2 fine_2_turbine_centred.geo
	gmsh -2 fine_15_turbine.geo

clean:
	rm -Rf steady/outputs/*
