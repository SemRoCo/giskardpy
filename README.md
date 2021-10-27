# giskardpy
The core python library of the Giskard framework for constraint- and optimization-based robot motion control.

## Installation instructions. Tested with Ubuntu 18.04 + melodic and 20.04 + Noetic

Install the following python packages. When using 20.04, just install the latest version of everything and use pip3:
```
sudo pip install pybullet
sudo pip install scipy==1.2.2 # this is the last version for python 2.7
sudo pip install casadi
sudo pip install sortedcontainers
sudo pip install hypothesis==4.34.0 # only needed if you want to run tests
sudo pip install pandas==0.24.2
sudo pip install numpy==1.16
sudo apt install python3-dev 
```
Install one of the following QP solver. The solvers are ordered by how fast they can solve the problem constructed by Giskard. QPOases is the fastest opensource solver for my usecase, that I have found. However, it is still significantly slower than the other two options:
 - Gurobi:
   - ```sudo pip3 install gurobipy```
   - If you have vpn access or are in the local network of the IAI of the University of Bremen, follow these instructions: https://ai.uni-bremen.de/wiki/intern/adm/gurobi
   - Otherwise you can apply for a free academic license or buy a license here: https://www.gurobi.com/academia/academic-program-and-licenses/
 - cplex 
   - Check this PR for more information: https://github.com/SemRoCo/giskardpy/pull/77
 - QPOases 
   - Install noetic branch of https://github.com/SemRoCo/qpOASES

Finally, you will need to set the ```qp_solver``` name in the config file, you are loading. E.g. here: https://github.com/SemRoCo/giskardpy/blob/mpc/config/default.yaml.

Now create the workspace
```
source /opt/ros/<ros-version>/setup.bash    # source ROS
mkdir -p ~/giskardpy_ws/src                 # create directory for workspace
cd ~/giskardpy_ws                           # go to workspace directory
catkin init                                 # init workspace, you might have to pip install catkin-tools
cd src                                      # go to source directory of workspace
wstool init                                 # init rosinstall
wstool merge https://raw.githubusercontent.com/SemRoCo/giskardpy/devel/rosinstall/<ros-version>.rosinstall
                                            # update rosinstall file
wstool update                               # pull source repositories
rosdep install --ignore-src --from-paths .  # install dependencies available through apt
cd ..                                       # go to workspace directory
catkin build                                # build packages
source ~/giskardpy_ws/devel/setup.bash      # source new overlay
```

## Fast Custom Bullet Bindings
Giskard will run much faster with Adrian Röfers bullet bindings instead of the official ones.
```
./scripts/build_pybullet.sh /my/awesome/library/dir

```
If everything worked fine, you should be able to do:
```python
import betterpybullet as bpb
```

### Tutorials
http://giskard.de/wiki:tutorials

