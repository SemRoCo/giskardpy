ARG BASE_IMAGE=ubuntu:focal
FROM ${BASE_IMAGE}
ARG ROS_PKG=desktop_full
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
            ros-noetic-desktop \
            ros-noetic-desktop-full \
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
#######
#RUN mkdir ros_catkin_ws && \
#    cd ros_catkin_ws && \
#    rosinstall_generator ${ROS_PKG} giskardpy --rosdistro noetic --deps --tar > ${ROS_DISTRO}-${ROS_PKG}.rosinstall && \
#    mkdir ./src && \
#    vcs import --input noetic-desktop_full.rosinstall ./src && \
#    apt-get update && \
#    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro ${ROS_DISTRO} --skip-keys python3-pykdl -y && \
#   python3 ./src/catkin/bin/catkin_make_isolated --install --install-space ${ROS_ROOT} -DCMAKE_BUILD_TYPE=Release && \
#    rm -rf /var/lib/apt/lists/*
##########################
RUN mkdir -p ~/catkin_ws/src 
RUN cd ~/catkin_ws/
RUN catkin init
RUN cd src                              
RUN wstool init 
RUN wstool merge https://raw.githubusercontent.com/Alok018/giskardpy/noetic-devel/rosinstall/catkin.rosinstall 
                                            
RUN wstool update 
RUN rosdep install --ignore-src --from-paths . 
RUN cd ..                                      
RUN catkin build                      

RUN echo 'source ${ROS_ROOT}/setup.bash' >> /root/.bashrc
WORKDIR /

