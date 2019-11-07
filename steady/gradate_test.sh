#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'Please supply gradation labels and sizes.'
elif [[ $# -eq 1 ]] ; then
    echo 'Please supply gradation sizes.'
else
    python3 test.py -dm_prescribed_boundary_labels $1 -dm_prescribed_boundary_sizes $2
    #scp outputs/hessian/mesh_0.vtu jgw116@ese-richter.ese.ic.ac.uk:~/src/turbine/plots_from_richter/elev.vtu
fi
