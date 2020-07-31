# giskardpy
The core python library of the Giskard framework for constraint- and optimization-based robot motion control.

## Installation instructions. Tested with Ubuntu 16.04 + ROS kinetic and 18.04 + melodic

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
source /opt/ros/kinetic/setup.bash          # start using ROS kinetic. Replace with melodic, if you are using it.
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

### (optional) symengine
If you want to use the symengine backend, it can be install like this (does not have to be part of the ros workspace):
```
sudo apt-get install llvm-6.0-dev # or llvm-8-dev if you are using 18.04
git clone https://github.com/symengine/symengine.git
git clone https://github.com/symengine/symengine.py.git
cd symengine
git checkout `cat ../symengine.py/symengine_version.txt`
mkdir build
cd build
cmake -DWITH_LLVM:BOOL=ON -DBUILD_TESTS:BOOL=OFF -DBUILD_BENCHMARKS:BOOL=OFF ..
make
sudo make install
cd ../../symengine.py
sudo python setup.py install
```

### Tutorials
http://giskard.de/wiki:tutorials

