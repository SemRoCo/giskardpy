# giskardpy
The core python library of the Giskard framework for constraint- and optimization-based robot motion control.

## Installation instructions for 16.04/kinetic

First install symengine + symengine.py
```
sudo apt-get install llvm-6.0-dev
git clone https://github.com/symengine/symengine.git
git clone https://github.com/symengine/symengine.py.git
cd symengine
git checkout `cat ../symengine.py/symengine_version.txt`
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DWITH_LLVM:BOOL=ON .
make
sudo make install
cd ../symengine.py
sudo python setup.py install
```

Install pybullet:
```
sudo pip install pybullet
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

### Tutorials
http://giskard.de/wiki:tutorials
