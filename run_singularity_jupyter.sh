singularity exec --nv --home $PWD --bind $DATAPATH \
  --env PYTHONPATH="\$PYTHONPATH:${PWD}/src",DATAPATH=$DATAPATH \
  external/containers/wave/wave_lib-cuda10.0_sep.sif \
  jupyter-lab --no-browser --port=6520 --ip 0.0.0.0