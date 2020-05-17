FROM continuumio/miniconda3 as build-wheel

RUN apt-get update && apt-get install -y -q git build-essential openmpi-bin libopenmpi-dev 
RUN conda install numpy nomkl ninja pyyaml setuptools cmake cffi
RUN conda install -c conda-forge openmpi

WORKDIR /code/
RUN git clone --branch v1.4.0 --depth 1 --recursive --shallow-submodules https://github.com/pytorch/pytorch 
ENV PYTORCH_BUILD_VERSION=1.4.0
ENV PYTORCH_BUILD_NUMBER=1
ENV USE_MPI=ON
ENV USE_MKLDNN=OFF
ENV USE_CUDA=OFF

ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN cd pytorch && python setup.py bdist_wheel

FROM python:3.7.7-slim-buster as ssh

RUN apt-get update && apt-get install -y -q openmpi-bin openmpi-common libopenmpi3 libopenmpi-dev libopenblas-dev libatlas-base-dev openssh-server

WORKDIR /code
COPY --from=build-wheel /code/pytorch/dist/torch-1.4.0-cp37-cp37m-linux_x86_64.whl .
RUN pip install torch-1.4.0-cp37-cp37m-linux_x86_64.whl
RUN pip install torchvision==0.5.0

RUN groupadd --gid 1000 pi
RUN useradd --system --create-home --shell /bin/bash --uid 1000 --gid pi --groups sudo pi
RUN echo 'pi:raspberry' | chpasswd
COPY --chown=pi:pi id_rsa.pub /home/pi/.ssh/authorized_keys
COPY --chown=pi:pi id_rsa /home/pi/.ssh/id_rsa

RUN mkdir -p /var/run/sshd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]