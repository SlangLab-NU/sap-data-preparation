#!/bin/bash

export PYTHONPATH=""
export apptainer_image=sap_data_prep_nemo.sif

apptainer shell --nv \
        --bind /scratch/lewis.jor:/scratch/lewis.jor \
        $apptainer_image
