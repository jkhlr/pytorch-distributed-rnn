FROM debian:buster-slim 

RUN apt-get update && apt-get install -y -q wget git build-essential openmpi-bin libopenmpi-dev libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools python3-pip

WORKDIR /var/src/pytorch
RUN git clone --branch v1.4.0 --depth 1 --recursive --shallow-submodules https://github.com/pytorch/pytorch .

ENV PYTORCH_BUILD_VERSION=1.4.0
ENV PYTORCH_BUILD_NUMBER=1
ENV USE_MPI=ON
ENV USE_MKLDNN=OFF
ENV USE_CUDA=OFF

RUN python3 setup.py bdist_wheel