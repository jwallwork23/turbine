all: mesh outdirs

mesh:
	gmsh -2 channel.geo

outdirs:
	mkdir outputs
	mkdir outputs-noid
	mkdir outputs/FixedMesh
	mkdir outputs/FixedMesh/hdf5
	mkdir outputs/FixedMesh/data
	mkdir outputs/HessianBased
	mkdir outputs/HessianBased/hdf5
	mkdir outputs/HessianBased/data
	mkdir outputs/Vorticity
	mkdir outputs/Vorticity/hdf5
	mkdir outputs/Vorticity/data
	mkdir outputs/DWP
	mkdir outputs/DWP/hdf5
	mkdir outputs/DWP/data
	mkdir outputs/DWR
	mkdir outputs/DWR/data
	mkdir outputs/DWR/hdf5
	mkdir outputs/AdjointOnly

clean:
	rm -Rf outputs
