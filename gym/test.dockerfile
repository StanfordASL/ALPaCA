# A Dockerfile that sets up a full Gym install with test dependencies
FROM ubuntu:16.04

# Install keyboard-configuration separately to avoid travis hanging waiting for keyboard selection
RUN \
    apt-get -y update && \
    apt-get install -y keyboard-configuration && \

# Maybe Install python3.6 on ubuntu 16.04 ?
#    apt-get install -y software-properties-common && \
#    add-apt-repository -y ppa:jonathonf/python-3.6 && \
#    apt-get -y update && \
#    apt-get -y install python3.6 python3.6-distutils python3.6-dev

    apt-get install -y \ 
        python-setuptools \
        python-pip \
        libpq-dev \
        zlib1g-dev \
        libjpeg-dev \
        curl \
        cmake \
        swig \
        python-opengl \
        python-numpy \
        python-pyglet \
        python3-opengl \
        libboost-all-dev \
        libsdl2-dev \
        libosmesa6-dev \
        patchelf \
        wget \
        unzip \
        git \
        vim \
        xvfb \
        ffmpeg \
        python3-dev && \

    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install tox && \

# Download mujoco
    mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    wget https://www.roboti.us/download/mjpro150_linux.zip  && \
    unzip mjpro150_linux.zip && \

# setup vim to be humane and compatible with codebase standards
    echo "set expandtab number shiftwidth=4 tabstop=4" > /root/.vimrc

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin

# install dependencies
COPY . /usr/local/gym/
RUN cd /usr/local/gym && \
    tox --notest 

WORKDIR /usr/local/gym/
ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
CMD ["tox"]
