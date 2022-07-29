singularity exec --nv --home $PWD --bind $DATAPATH \
  --env DATAPATH=$DATAPATH \
  external/containers/pyseis/pyseis-wave_lib_cuda10.0_sep.sif \
  jupyter-lab --no-browser --port=6520 --ip 0.0.0.0