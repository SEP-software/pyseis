FROM sfarris1994/wave_lib:cuda10.0_sep

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update

RUN python -m pip install -U pip wheel setuptools 

RUN git clone http://cees-gitlab.stanford.edu/sfarris/pyseis.git /opt/pyseis &&\
  cd /opt/pyseis 

RUN cd /opt/pyseis && pip install -e . 

RUN cd /opt/pyseis && pytest -m "not gpu"