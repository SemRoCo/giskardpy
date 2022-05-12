# giskardpy
The core python library of the Giskard framework for constraint- and optimization-based robot motion control.

## Installation instructions. Tested with Ubuntu 18.04 + melodic and 20.04 + Noetic

Install the following python packages. When using 20.04, just install the latest version of everything:
```
sudo pip install pybullet==3.0.8 # last known version to work with python 2.7
sudo pip install scipy==1.2.2 # this is the last version for python 2.7
sudo pip install casadi
sudo pip install sortedcontainers
sudo pip install hypothesis==4.34.0 # only needed if you want to run tests
sudo pip install pandas==0.24.2
sudo pip install numpy==1.16
```

Now create the workspace
```
source /opt/ros/<ros-version>/setup.bash    # source ROS
mkdir -p ~/giskardpy_ws/src                 # create directory for workspace
cd ~/giskardpy_ws                           # go to workspace directory
catkin init                                 # init workspace, you might have to pip install catkin-tools
cd src                                      # go to source directory of workspace
wstool init                                 # init rosinstall
wstool merge https://raw.githubusercontent.com/SemRoCo/giskardpy/master/rosinstall/<ros-version>.rosinstall
                                            # update rosinstall file
wstool update                               # pull source repositories
rosdep install --ignore-src --from-paths .  # install dependencies available through apt
cd ..                                       # go to workspace directory
catkin build                                # build packages
source ~/giskardpy_ws/devel/setup.bash      # source new overlay
```

### Tutorials
http://giskard.de/wiki:tutorials

