# make synthetic seismic data 

## Software and hardware requirements
* At least one NVIDIA gpu with CUDA Version >= 10.0
Download and install the following applications:
## Getting started
1. clone the repo.<br>
  `$ git clone http://cees-gitlab.stanford.edu/sfarris/pyseis.git && cd pyseis`
2. Run notebooks using jupyter lab from within docker container. Container has all required packages and compiled C++ binaries.
```console
sudo docker run -p 7001:7001 \
  --gpus=all \
  -e LOCAL_USER_ID=`id -u $USER` \
  -e DATAPATH=$DATAPATH \
  -v $DATAPATH:$DATAPATH \
  -v $(pwd):$(pwd) \
  -w $(pwd) \
  sfarris1994/pysynth:pyseis_wave_lib_v1 jupyter-lab --no-browser --port=7001 --ip 0.0.0.0
```
