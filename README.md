# Giskard(py)
Giskard is an open source motion planning framework for ROS, which uses constraint and optimization based task space control to generate trajectories for the whole body of mobile manipulators.

## Installation instructions for Ubuntu 20.04 + Noetic

#### Python dependencies
```
pip3 install -r pandas numpy==1.23.4 hypothesis qpalm
```

#### Workspace
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

#### Fast Custom Bullet Bindings
Giskard uses Adrian Röfer's bullet bindings (https://github.com/ARoefer/bullet3) instead of the official ones, as they are much faster for our use case.
Use this script to install them:
```
./scripts/build_better_pybullet.sh /path/of/your/choosing
source ~/.bashrc
```
Where `/path/of/your/choosing` can be for example a new folder in your home directory.
If everything worked fine, you should be able to do the following without any errors:
```
$ ipython3
Python 3.8.2 (default, Mar 13 2020, 10:14:16) 
Type 'copyright', 'credits' or 'license' for more information
IPython 8.1.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import betterpybullet

In [2]:
```
If it doesn't work, source your ```.bashrc``` again 
and/or make sure that your ```$PYTHONPATH``` includes something like ```/path/of/your/choosing/bullet3/build_cmake/better_python:/path/of/your/choosing/bullet3/examples/pybullet```. 

#### QP solvers
Giskard supports multiple QP solvers and will automatically use the fasted installed solver.
The default is `qpalm`, as it is the easiest to install.

You may want to install `qpSWIFT` manually, as it is a little faster than `qpalm`.
To do so follow the instructions on their github page: https://github.com/qpSWIFT/qpSWIFT.

Giskard also supports three additional solvers, which are slower than `qpalm`, sorted by how fast they are for Giskard's usecase:
- `Gurobi`: A commercial solver, which is approximately as fast as `qpalm`: 
  - ```sudo pip3 install gurobipy```
  - You can apply for a free academic license or buy one here: https://www.gurobi.com/academia/academic-program-and-licenses/
- `Clarabel.rs`: `sudo pip3 install clarabel` (https://github.com/oxfordcontrol/Clarabel.rs)
- `qpOASES`: https://github.com/SemRoCo/qpOASES/tree/noetic

### Tutorials
https://github.com/SemRoCo/giskardpy/wiki

### How to cite
Stelter, Simon, Georg Bartels, and Michael Beetz. "An open-source motion planning framework for mobile manipulators using constraint-based task space control with linear MPC." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.

