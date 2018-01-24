# giskardpy
The core python library of the Giskard framework for constraint- and optimization-based robot motion control.

## Installation instructions for 16.04/kinetic

First install symengine + symengine.py
```
git clone https://github.com/symengine/symengine.git
cd symengine
git checkout 2f5ff9db9ff511ee243438a85ea8e2da2d05af39
cmake -DWITH_LLVM:BOOL=ON .
make
make install
cd ..
git clone https://github.com/symengine/symengine.py.git
cd symengine.py
sudo python setup.py install
```

Now create the workspace
```
source /opt/ros/kinetic/setup.bash          # start using ROS kinetic
mkdir -p ~/giskardpy_ws/src                 # create directory for workspace
cd ~/giskardpy_ws                           # go to workspace directory
catkin init                                 # init workspace
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

### Tests
Run
```
catkin build --catkin-make-args run_tests  # build packages
```