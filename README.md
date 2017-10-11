# giskardpy
The core python library of the Giskard framework for constraint- and optimization-based robot motion control.

## Installation instructions for 16.04/kinetic

First install sympy using pip because python-sympy, which can be installed using apt, is outdated.
```
sudo pip install sympy
```

Now create the workspace
```
source /opt/ros/kinetic/setup.bash         	# start using ROS kinetic
mkdir -p ~/giskardpy_ws/src                 # create directory for workspace
cd ~/giskardpy_ws                           # go to workspace directory
catkin init                               	# init workspace
cd src                                    	# go to source directory of workspace
rosdep install --ignore-src --from-paths .	# install dependencies available through apt
cd ..                                      	# go to workspace directory
catkin build 							   	# build packages
source ~/giskardpy_ws/devel/setup.bash 	    # source new overlay
```

### Tests
Run
```
catkin build --catkin-make-args run_tests  # build packages
```