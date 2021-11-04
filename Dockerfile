ARG BASE_IMAGE=ubuntu:focal
FROM ${BASE_IMAGE}
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive
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

ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p $CATKIN_WS/src
WORKDIR $CATKIN_WS/src

# Initialize local catkin workspace
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && apt-get update \
    # ROS File Server
    && git clone --branch noetic-devel https://github.com/Alok018/giskardpy.git \
    && git clone --branch devel https://github.com/SemRoCo/giskard_msgs.git \
    && git clone --branch noetic https://github.com/SemRoCo/qpOASES.git \
    && git clone https://github.com/code-iai/omni_pose_follower.git \
    # Install dependencies
    && cd $CATKIN_WS \
    && rosdep install -y --from-paths . --ignore-src --rosdistro ${ROS_DISTRO} \
    # Build catkin workspace
    && catkin_make

# Always source ros_catkin_entrypoint.sh when launching bash (e.g. when attaching to container)
RUN echo "source /ros_catkin_entrypoint.sh" >> /root/.bashrc

