FROM ubuntu:20.04

ARG DOWNLOAD_URL

SHELL ["/bin/bash", "-c"]

# Install Dependencies, tzdata install noninteractive, because of Timezone setup
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt-get -y install tzdata
RUN apt-get install -y curl python3 python3-pip git-all cmake sudo pkg-config python3-numpy

# Get requirements.txt from DOWNLOAD_URL-Parameter
RUN curl -fsSL $DOWNLOAD_URL -o requirements.txt

RUN sudo apt install libeigen3-dev && \
    python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt

# Install qpSWIFT
RUN cd ~ && \
    git clone https://github.com/SemRoCo/qpSWIFT && \
    cd qpSWIFT && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release && \
    cmake --build build --target install && \
    cd python && \
    python3 setup.py install

# Install betterpybullet
RUN mkdir -p ~/bpb && \
    cd ~/bpb && \
    git clone https://github.com/SemRoCo/bullet3.git && \
    cd bullet3 && \
    ./build_better_pybullet.sh
