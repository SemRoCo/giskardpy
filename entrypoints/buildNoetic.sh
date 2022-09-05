

# build giskardpy
source /opt/ros/noetic/setup.bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws
catkin init
cd src
wstool init
wstool merge https://raw.githubusercontent.com/SemRoCo/giskardpy/master/rosinstall/catkin.rosinstall
wstool update
cd giskardpy
git checkout devel
cd ..
rosdep install -y --rosdistro noetic --ignore-src --from-paths .
cd ..
catkin build

# build pr2 simulator
source /opt/ros/noetic/setup.bash
cd ~/ros_ws/src
wstool merge https://raw.githubusercontent.com/code-iai/iai_pr2/master/iai_pr2_sim/rosinstall/catkin-noetic.rosinstall
wstool update
rosdep install -y --rosdistro noetic --ignore-src --from-paths iai_pr2/
git clone https://github.com/code-iai/iai_maps
git clone https://github.com/code-iai/iai_common_msgs
cd ..
catkin build
source devel/setup.bash