FROM python:3.7.3
RUN apt-get update -yq && apt-get install -yq build-essential zip unzip openjdk-8-jdk curl psmisc nodejs
RUN pip install cython==0.29.0 numpy

WORKDIR /var/src
RUN wget https://github.com/bazelbuild/bazel/releases/download/2.1.1/bazel-2.1.1-dist.zip
RUN unzip -d bazel bazel-2.1.1-dist.zip
WORKDIR /var/src/bazel
RUN env BAZEL_JAVAC_OPTS="-J-Xmx2g -J-Xms2g" EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
RUN cp output/bazel /usr/bin/

# RUN curl -sL https://deb.nodesource.com/setup_11.x  | bash -

WORKDIR /var/src/ray
RUN git clone https://github.com/ray-project/ray.git .
RUN git checkout ray-0.8.6

WORKDIR /var/src/ray/python/ray/dashboard/client
RUN npm ci
RUN npm run build

WORKDIR /var/src/ray/python
RUN python setup.py bdist_wheel

WORKDIR /var/src/rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /var/src/py-spy
RUN git clone https://github.com/benfred/py-spy.git .
RUN git checkout v0.3.3
RUN python setup.py bdist_wheel
