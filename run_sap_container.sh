
#!/bin/bash


export PYTHONPATH=""
export apptainer_image=sap_data_prep.sif

export CACHE_ROOT=/scratch/lewis.jor/cache

export APPTAINERENV_XDG_CACHE_HOME=$CACHE_ROOT

export APPTAINERENV_NLTK_DATA=$CACHE_ROOT/nltk
export APPTAINERENV_HF_HOME=$CACHE_ROOT/huggingface
export APPTAINERENV_TRANSFORMERS_CACHE=$CACHE_ROOT/huggingface
export APPTAINERENV_MPLCONFIGDIR=$CACHE_ROOT/matplotlib
export APPTAINERENV_TORCH_HOME=$CACHE_ROOT/torch
export APPTAINERENV_CACHED_PATH_CACHE_ROOT=$CACHE_ROOT/cached_path

mkdir -p $CACHE_ROOT/nltk
mkdir -p $CACHE_ROOT/huggingface
mkdir -p $CACHE_ROOT/matplotlib
mkdir -p $CACHE_ROOT/torch
mkdir -p $CACHE_ROOT/cached_path

apptainer shell --nv \
        --no-home \
       	--bind /scratch/lewis.jor:/scratch/lewis.jor \
       	$apptainer_image

