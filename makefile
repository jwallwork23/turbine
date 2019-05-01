all: mesh

mesh:
	gmsh -2 channel.geo

clean:
	rm -Rf steady/outputs/*
