FROM ubuntu:24.04

ARG DOWNLOAD_URL

SHELL ["/bin/bash", "--login", "-c"]

ENV WORKON_HOME=/root/venvs

# Installing Dependencies
# tzdata install noninteractive, because of Timezone setup
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt-get -y install tzdata
RUN apt-get install -y curl python3 python3-pip python3-venv git-all cmake \
            sudo pkg-config python3-virtualenvwrapper python3-numpy

# Install virtualenvwrapper \
# and create giskardpy venv
RUN mkdir -p $WORKON_HOME && \
    echo "export WORKON_HOME=$WORKON_HOME" >> ~/.bashrc && \
    echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> ~/.bashrc
# mkvirtualenv returns exit 1, but works fine \
# so overwriting with exit 0, so this step doesn't fail
RUN mkvirtualenv giskardpy --system-site-packages; exit 0
RUN echo "source /root/venvs/giskardpy/bin/activate" >> ~/.bashrc

# Download python-requirements
# and install in venv
RUN curl -fsSL $DOWNLOAD_URL -o requirements.txt
RUN workon giskardpy && \
    sudo apt-get install libeigen3-dev && \
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
    git clone -b jazzy https://github.com/SemRoCo/bullet3.git && \
    cd bullet3 && \
    ./build_better_pybullet.sh

# Download PR2-Repo and compile urdf
RUN workon giskardpy && \
    cd ~ && \
    pip3 install xacro && \
    git clone https://github.com/code-iai/iai_pr2 && \
    sed -i 's|$(find iai_pr2_description)|/root/iai_pr2/iai_pr2_description|g' ~/iai_pr2/iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro && \
    xacro ~/iai_pr2/iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro > pr2.urdf && \
    sed -i 's|package://|/root/iai_pr2/|g' pr2.urdf && \
    sed -i 's|/pr2_description/|/iai_pr2_description/|g' pr2.urdf