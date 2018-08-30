all: outdirs

outdirs:
	mkdir outputs
	mkdir outputs/FixedMesh
	mkdir outputs/HessianBased
	mkdir outputs/DWP
	mkdir outputs/DWR
	mkdir outputs/DWR/hdf5

clean:
	rm -Rf outputs
