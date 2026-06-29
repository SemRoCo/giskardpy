# This Library has been moved!

[The current version is available here](https://github.com/cram2/cognitive_robot_abstract_machine/tree/main/giskardpy)

# Giskardpy
Giskardpy is an open source library for implementing motion control frameworks.
It uses constraint and optimization based task space control to control the whole body of mobile manipulators.

This is a pure python library with the core functionality.
To use it with ROS you need the following repos, which use giskardpy to create an action server and implement ROS interfaces:
 - **ROS1**: https://github.com/SemRoCo/giskardpy_ros/tree/ros1-noetic-main
 - **ROS2**: https://github.com/SemRoCo/giskardpy_ros/tree/ros2-jazzy-main

## Installation instructions for Ubuntu (tested on 20.04 and 24.04)

### (Optional) create a virtual environment using virtualenvwrapper
```
sudo apt install virtualenvwrapper
echo "export WORKON_HOME=~/venvs" >> ~/.bashrc
echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
mkdir -p $WORKON_HOME

# --system-site-packages is only required if you are using ROS
mkvirtualenv giskardpy --system-site-packages
```
To use it do:
```
workon giskardpy
```

### Build Giskardpy
Switch to your venv, if you use one.
```
workon giskardpy
```
Choose a place where you want to build giskardpy and clone it. This should NOT be in a ROS workspace.
```
mkdir -p ~/libs && cd ~/libs
git clone -b giskard_library https://github.com/SemRoCo/giskardpy.git
cd giskardpy
```
Install Giskardpy, `-e` is optional but prevents you from having to rebuild every time the code changes.
```
pip3 install -r requirements.txt
pip3 install -e .                           
```

#### (Optional) Install alternative QP solvers
Giskard supports alternative QP solvers, but they are all slower than qpSWIFT, which is the default.

- `gurobi`: Commercial solver. Useful for debugging during development.
  - ```sudo pip3 install gurobipy```
  - You can apply for a free academic license or buy one here: https://www.gurobi.com/academia/academic-program-and-licenses/
  - If you have vpn access to or are in the local network of the IAI of the University of Bremen, follow these instructions: https://ai.uni-bremen.de/wiki/intern/adm/gurobi

[//]: # (- `Clarabel.rs`: `sudo pip3 install clarabel` &#40;https://github.com/oxfordcontrol/Clarabel.rs&#41;)


### (Optional) Build Custom Bullet Bindings
Giskard uses Adrian Röfer's bullet bindings instead of the official ones, as they are much faster for our use case.
Install them like this:
```
workon giskardpy
mkdir -p ~/libs && cd ~/libs                # choose a place where you want to build pybullet
git clone https://github.com/SemRoCo/bullet3.git
pip install .
```
To test your installation do:
```
workon giskardpy
python3 -c "import betterpybullet"
```
If it doesn't work, make sure that your ```$PYTHONPATH``` includes something like 
```/path/to/your/bullet3/build_cmake/better_python:/path/to/your/bullet3/examples/pybullet```. 


### Tutorials
https://github.com/SemRoCo/giskardpy/wiki

Try it out in our Virtual Research building: https://vrb.ease-crc.org/explore-labs/giskard-lab/

### How to cite
```
@phdthesis{stelter25giskard,
	author = {Simon Stelter},
	title = {A Robot-Agnostic Kinematic Control Framework: Task Composition via Motion Statecharts and Linear Model Predictive Control},
	year = {2025},
	doi = {10.26092/elib/3743},	
}
```
