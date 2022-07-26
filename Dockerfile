From  nvidia/cuda:10.0-devel-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update 
RUN    apt-get -y  install g++ git make cmake gcc 
RUN apt-get -y install libtbb-dev 
RUN apt-get -y  install g++  git make gcc libboost-all-dev  libboost-dev wget libssl-dev
RUN apt-get -y install  cmake 
RUN apt-get -y install libboost-all-dev  libboost-dev
RUN   apt-get -y install  cmake libtbb-dev
RUN  apt-get -y  install  libssl-dev flex

RUN cd /opt/ &&\
  wget  https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh  &&\
  bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -p /opt/conda\
  && rm -f Miniconda3-py37_4.8.2-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install scikit-build numpy


RUN cd /tmp &&\
  wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1.tar.gz &&\
  cd /tmp/ &&\
  tar xzf /tmp/cmake-3.22.1.tar.gz &&\
  cd cmake-3.22.1 &&\
  cmake . &&\
  make -j 12  && \
  make -j 12 install

RUN apt-get -y install openjdk-8-jdk gfortran


RUN conda install numba matplotlib numpy=1.21 scipy jpype1 jupyterlab h5py gpustat pytest 
RUN conda install dask distributed dask-jobqueue -c conda-forge


RUN  apt-get -y  install g++ python3-numpy git make gcc 
RUN apt-get -y install libboost-all-dev  libboost-dev
RUN   apt-get -y install  cmake python3-dev python3-pytest python3-numpy-dbg libtbb-dev
RUN  apt-get -y  install gfortran libfftw3-3 libfftw3-dev python3-pip libssl-dev
RUN pip3 install scikit-build

RUN apt-get -y install flex libxaw7-dev  

RUN git clone http://zapad.Stanford.EDU/bob/SEPlib.git  /opt/sep-main/src && \
  cd /opt/sep-main/src &&\
  git checkout b106d7f0f33be36b4e19b91095def70e40981307 &&\
  mkdir /opt/sep-main/build &&\
  cd /opt/sep-main/build &&\
  cmake  -DCMAKE_INSTALL_PREFIX=/opt/SEP ../src &&\
  make  -j 12  install

RUN python --version
RUN git clone http://cees-gitlab.stanford.edu/sfarris/wave_lib.git /opt/wave_lib 
RUN cd /opt/wave_lib &&\
  git submodule update --init --recursive -- src/external/pybind11 &&\
  git submodule update --init --recursive -- src/external/python-solver &&\
  git submodule update --init --recursive -- src/external/sep-iolibs &&\
  mkdir -p /opt/wave_lib/build &&\
  cd /opt/wave_lib/build &&\
  cmake  -DCMAKE_INSTALL_PREFIX=../local -DCMAKE_CXX_FLAGS="-O3 -fpermissive" -DBUILD_SHARED_LIBS=True ../src &&\  
  make -j 16 install &&\
  rm -rf *
RUN chmod -R 777 /opt/wave_lib/local/bin

ENV PYTHONPATH="/opt/wave_lib/local/lib/python3.7:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/opt/wave_lib/local/lib:${LD_LIBRARY_PATH}"

RUN python -c 'import Acoustic_iso_float'
RUN python -c 'import interpBSplineModule'
RUN python -c 'import Acoustic_iso_float_3D'
RUN python -c 'import interpBSplineModule_3D'
RUN python -c 'import genericIO; import pyNonLinearSolver; from Acoustic_iso_float import nonlinearPropShotsGpu;'
RUN python -c 'import Acoustic_iso_float; import Acoustic_iso_float_3D;import Elastic_iso_float_prop; import Elastic_iso_float_3D' 


ENV PYTHONPATH="${PYTHONPATH}:/opt/SEP/lib/python3.7"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/SEP/lib:/opt/SEP/lib/python3.7"
ENV PATH="/opt/SEP/bin:${PATH}"

RUN python -c 'import genericIO; import pyNonLinearSolver; from Acoustic_iso_float import nonlinearPropShotsGpu;'
RUN python -c 'import Acoustic_iso_float; import Acoustic_iso_float_3D;import Elastic_iso_float_prop; import Elastic_iso_float_3D' 

RUN pip install mock