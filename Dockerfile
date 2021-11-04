ARG BASE_IMAGE=ubuntu:focal
FROM ${BASE_IMAGE}

ARG ROS_PKG=ros_base
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# add the ROS deb repo to the apt sources list
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install bootstrap dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpython3-dev \
    python3-rosdep \
    python3-pip \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    python3-vcstool \
    ros-noetic-urdfdom-py \
    ros-noetic-py-trees \
    ros-noetic-py-trees-ros \
    ros-noetic-catkin \
    build-essential && \
    rosdep init && \
    rosdep update && \
    rm -rf /var/lib/apt/lists/*

COPY dependencies.txt dependencies.txt
RUN pip install -r dependencies.txt  

RUN mkdir ros_catkin_ws && \
    cd ros_catkin_ws && \
    mkdir src && \
    git clone --branch noetic-devel https://github.com/Alok018/giskardpy.git && \
    git clone --branch devel https://github.com/SemRoCo/giskard_msgs.git && \
    git clone --branch noetic https://github.com/SemRoCo/qpOASES.git && \
    git clone https://github.com/code-iai/omni_pose_follower.git && \
    vcs import --input ${ROS_DISTRO}-${ROS_PKG}.rosinstall ./src && \
    apt-get update && \
    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro ${ROS_DISTRO} -y && \
    python3 ./src/catkin/bin/catkin_make_isolated --install --install-space ${ROS_ROOT} -DCMAKE_BUILD_TYPE=Release && \
    rm -rf /var/lib/apt/lists/*

RUN echo 'source ${ROS_ROOT}/setup.bash' >> /root/.bashrc
WORKDIR /
