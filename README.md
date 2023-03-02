# GPU accelerated wave equation package for modeling seismic data, imaging the earth's interior, and inverting for earth parameters. 

Uses:
- Model seismic experiments with various wave equations.
- Run wave-equation based imaging e.g. reverse-time migration (RTM) and least squares RTM.
- Invert for earth model parameters using the wave equation e.g. full waveform inversion

Available wave equations:
- Acoustic, isotropic, constant density in 2D and 3D
- Elastic, isotropic in 2D and 3D (velocity/stress formulation)

Why it’s useful:
- GPU accelerated
- Intuitive python interface

## Software and hardware requirements
* At least one NVIDIA gpu with CUDA Version >= 10.0
* [Docker](https://docs.docker.com/engine/install/)

## Getting started
1. clone the repo.
```console
git clone http://cees-gitlab.stanford.edu/sfarris/pyseis.git && cd pyseis
```
2. Run notebooks using jupyter lab from within docker container. Container has all required packages and compiled C++ binaries.
```console
sudo docker run -p 7001:7001 \
  --gpus=all \
  -e LOCAL_USER_ID=`id -u $USER` \
  -v $(pwd):$(pwd) \
  -w $(pwd) \
  sfarris1994/pyseis:stable jupyter-lab --no-browser --port=7001 --ip 0.0.0.0
```
3. Navigate to the jupyterlab at `http://127.0.0.1:7001` in your browser.

## About the repo
* This goal of this repository is to make wave equation modeling, imaging, and inversion more accesible for researchers.
* Authors for this repository: [Stuart Farris](https://www.linkedin.com/in/stuart-farris/) (sfarris@sep.stanford.edu), [Guillaume Barnier](https://gbarnier.github.io) (barnier@gmail.com), and [Ettore Biondi](https://www.linkedin.com/in/ettore-biondi/) (ebiondi@caltech.edu). 
* Date: 03/02/2023
* Feel free to contact us for any questions or bugs to report
