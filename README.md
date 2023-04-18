# Giskard(py)
Giskard is an open source motion planning framework for ROS, which uses constraint and optimization based task space control to generate trajectories for the whole body of mobile manipulators.

## Installation instructions for Ubuntu 20.04 + Noetic

#### ROS Workspace
```
source /opt/ros/noetic/setup.bash           # source ROS
mkdir -p ~/giskardpy_ws/src                 # create directory for workspace
cd ~/giskardpy_ws                           # go to workspace directory
catkin init                                 # init workspace, you might have to pip install catkin-tools
cd src                                      # go to source directory of workspace
wstool init                                 # init rosinstall
wstool merge https://raw.githubusercontent.com/SemRoCo/giskardpy/master/rosinstall/noetic.rosinstall
                                            # update rosinstall file
wstool update                               # pull source repositories
rosdep install --ignore-src --from-paths .  # install dependencies available through apt
pip3 install -r giskardpy/requirements.txt  # install python deps
cd ..                                       # go to workspace directory
catkin build                                # build packages
source ~/giskardpy_ws/devel/setup.bash      # source new overlay
```

#### Custom Bullet Bindings
Giskard uses Adrian Röfer's bullet bindings instead of the official ones, as they are much faster for our use case.
Install them like this:
```
mkdir -p ~/libs && cd ~/libs                # choose a place where you want to build pybullet
git clone https://github.com/SemRoCo/bullet3.git
cd bullet3                                  # be sure to be in the bullet3 folder
./build_better_pybullet.sh                  # this script will also clone and build pybind11 into libs
source ~/.bashrc                            # the script adds a python path modification to your bashrc
```
To test your installation do:
```
$ python3
Python 3.8.10 (default, Nov 14 2022, 12:59:47) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import betterpybullet
>>>
```
If it doesn't work, make sure that your ```$PYTHONPATH``` includes something like 
```/path/to/your/bullet3/build_cmake/better_python:/path/to/your/bullet3/examples/pybullet```. 

#### Alternative QP solvers
Giskard supports multiple QP solvers and will automatically use the fasted installed solver.

- `qpalm`: Default solver, because it is the easiest to install and still reasonably fast.
- `qpSWIFT`: Fastest open source solver in most cases. Install instructions: https://github.com/qpSWIFT/qpSWIFT.
- `gurobi`: Commercial solver. Slightly slower than `qpSWIFT` on most robots. Outperforms `qpSWFIT` on systems with a lot of dof and/or a large prediction horizon.
  - ```sudo pip3 install gurobipy```
  - You can apply for a free academic license or buy one here: https://www.gurobi.com/academia/academic-program-and-licenses/
  - If you have vpn access to or are in the local network of the IAI of the University of Bremen, follow these instructions: https://ai.uni-bremen.de/wiki/intern/adm/gurobi

[//]: # (- `Clarabel.rs`: `sudo pip3 install clarabel` &#40;https://github.com/oxfordcontrol/Clarabel.rs&#41;)


### Tutorials
https://github.com/SemRoCo/giskardpy/wiki

### How to cite
Stelter, Simon, Georg Bartels, and Michael Beetz. "An open-source motion planning framework for mobile manipulators using constraint-based task space control with linear MPC." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.

