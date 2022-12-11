# Giskard(py)
Giskard is an open source motion planning framework for ROS, which uses constraint and optimization based task space control to generate trajectories for the whole body of mobile manipulators.

## Installation instructions for Ubuntu 20.04 + Noetic

#### Python dependencies
```
sudo pip3 install pandas numpy==1.23.4 hypothesis
sudo apt install python3-dev 
```
#### Gurobi
This step is optional but recommanded because Gurobi is significantly faster than QPOases, but it requires a license.
If Gurobi is not installed, Giskard will use QPOases automatically as a backup.
   - ```sudo pip3 install gurobipy```
   - If you have vpn access or are in the local network of the IAI of the University of Bremen, follow these instructions: https://ai.uni-bremen.de/wiki/intern/adm/gurobi
   - Otherwise you can apply for a free academic license or buy one here: https://www.gurobi.com/academia/academic-program-and-licenses/

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
cd ..                                       # go to workspace directory
catkin build                                # build packages
source ~/giskardpy_ws/devel/setup.bash      # source new overlay
```

#### Fast Custom Bullet Bindings
Giskard uses Adrian Röfer's bullet bindings instead of the official ones, as they are much faster for our use case.
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

### Tutorials
https://github.com/SemRoCo/giskardpy/wiki

