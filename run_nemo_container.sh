#!/bin/bash

export PYTHONPATH=""
export PYTHONNOUSERSITE=1
export apptainer_image=sap_analysis.sif

apptainer shell --nv \
        --bind /scratch/lewis.jor:/scratch/lewis.jor \
        $apptainer_image
