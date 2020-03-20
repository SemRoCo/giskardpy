# giskardpy
The core python library of the Giskard framework for constraint- and optimization-based robot motion control.

## Installation instructions for 16.04/kinetic

Install the following python packages:
```
sudo pip install pybullet
sudo pip install scipy==1.2.2 # this is the last version for python 2.7
sudo pip install casadi
sudo pip install sortedcontainers
sudo pip install hypothesis # only needed if you want to run tests
sudo pip install pandas
```

Now create the workspace
```
source /opt/ros/kinetic/setup.bash          # start using ROS kinetic
mkdir -p ~/giskardpy_ws/src                 # create directory for workspace
cd ~/giskardpy_ws                           # go to workspace directory
catkin init                                 # init workspace, you might have to pip install catkin-tools
cd src                                      # go to source directory of workspace
wstool init                                 # init rosinstall
wstool merge https://raw.githubusercontent.com/SemRoCo/giskardpy/master/rosinstall/catkin.rosinstall
                                            # update rosinstall file
wstool update                               # pull source repositories
rosdep install --ignore-src --from-paths .  # install dependencies available through apt
cd ..                                       # go to workspace directory
catkin build                                # build packages
source ~/giskardpy_ws/devel/setup.bash      # source new overlay
```

### Tutorials
http://giskard.de/wiki:tutorials

