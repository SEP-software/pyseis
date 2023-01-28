# GPU accelerated wave equation package for modeling seismic data, imaging the earth's interior, and inverting for earth parameters. 

## Software and hardware requirements
* At least one NVIDIA gpu with CUDA Version >= 10.0
* [Docker](https://docs.docker.com/engine/install/)
## Getting started
1. clone the repo.<br>
  `$ git clone http://cees-gitlab.stanford.edu/sfarris/pyseis.git && cd pyseis`
2. Run notebooks using jupyter lab from within docker container. Container has all required packages and compiled C++ binaries.
```console
sudo docker run -p 7001:7001 \
  --gpus=all \
  -e LOCAL_USER_ID=`id -u $USER` \
  -v $(pwd):$(pwd) \
  -w $(pwd) \
  sfarris1994/pyseis:20230128-010323UTC-20a4d88 jupyter-lab --no-browser --port=7001 --ip 0.0.0.0
```
3. Navigate to the jupyterlab at `http://127.0.0.1:7001` in your browser.
