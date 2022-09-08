singularity shell --nv --home $PWD --bind $DATAPATH \
  --env DATAPATH=$DATAPATH \
  external/containers/pyseis/pyseis-wave_lib_cuda10.0_sep.sif