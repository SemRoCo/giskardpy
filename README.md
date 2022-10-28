# Giskard(py)
Giskard is an open source motion planning framework for ROS, which uses constraint and optimization based task space control to generate trajectories for the whole body of mobile manipulators.

## Installation instructions for Ubuntu 20.04 + Noetic

#### Python dependencies
```
sudo pip3 install scipy casadi sortedcontainers hypothesis pandas numpy trimesh colour pycollada
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
wstool merge https://raw.githubusercontent.com/SemRoCo/giskardpy/devel/rosinstall/noetic.rosinstall
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
Where `/path/of/your/choosing` can be e.g. a new folder in your home directory.
If everything worked fine, you should be able to do:
```python
import betterpybullet as bpb
```

### Tutorials
http://giskard.de/wiki:tutorials

