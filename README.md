# make synthetic seismic data 

## Software and hardware requirements
* At least one NVIDIA gpu with CUDA Version >= 10.0
Download and install the following applications:
* Singularity: https://sylabs.io/guides/3.0/user-guide/installation.html
* git lfs: https://github.com/git-lfs/git-lfs/wiki/Tutorial
## Getting started
1. clone the repo.<br>
  `$ git clone http://cees-gitlab.stanford.edu/sfarris/pyseis.git && cd pyseis`
2. initialize submodule.<br>
  `$ git submodule update --init -- external/containers` 
3. pull singularity image.<br>
  `$ cd external/containers && git lfs pull --include pysynth/pysynth-wave3d_cuda10.0_sep.sif --exclude "" && cd -` 
4. set your DATAPATH environment variable.<br>
  `$ export DATAPATH=<YOUR_DATA_PATH>`
5. run singularity shell.<br>
  `$ ./run_singularity_shell.sh`
