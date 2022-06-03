singularity exec --nv --home $PWD --bind $DATAPATH \
  --env PYTHONPATH="\$PYTHONPATH:${PWD}/src",DATAPATH=$DATAPATH \
  external/containers/pysynth/pysynth-wave3d_cuda10.0_sep.sif \
  jupyter-lab --no-browser --port=6520 --ip 0.0.0.0