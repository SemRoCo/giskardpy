FROM osrf/ros:noetic-desktop-full

ARG DOWNLOAD_URL

SHELL ["/bin/bash", "-c"]

# Install Dependencies, tzdata install noninteractive, because of Timezone setup
RUN apt-get update
RUN apt-get install -y curl \
            python3 \
            python3-pip \
            python3-wstool \
            python3-catkin-tools \
            python3-numpy \
            jq \
            wget \
            ros-noetic-py-trees-ros \
            ros-noetic-geometry2 \
            ros-noetic-tf2-bullet \
            ros-noetic-tf2-sensor-msgs \
            ros-noetic-tf2-tools \
            ros-noetic-urdfdom-py \
            ros-noetic-rqt-py-trees \
            ros-noetic-rospy-message-converter \
            ros-noetic-pr2-description \
            ros-noetic-effort-controllers \
            ros-noetic-map-server \
            ros-noetic-map-server \
            ros-noetic-joint-trajectory-controller \
            git-all

# Install gh cli for github workflow commands (can't use apt normally unlike ubuntu24.04)
RUN sudo mkdir -p -m 755 /etc/apt/keyrings \
    && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

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
