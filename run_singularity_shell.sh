singularity shell --nv --home $PWD --bind $DATAPATH \
  --env PYTHONPATH="\$PYTHONPATH:${PWD}/src",DATAPATH=$DATAPATH \
  external/containers/pysynth/pysynth-wave3d_cuda10.0_sep.sif