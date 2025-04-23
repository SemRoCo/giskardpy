# Giskardpy
Giskardpy is an open source library for implementing motion control frameworks.
It uses constraint and optimization based task space control to control the whole body of mobile manipulators.

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
Switch to your venv.
```
workon giskardpy
```
Choose a place where you want to build giskardpy and clone it
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

### (Optional) Build Custom Bullet Bindings
Giskard uses Adrian Röfer's bullet bindings instead of the official ones, as they are much faster for our use case.
Install them like this:
```
workon giskardpy
mkdir -p ~/libs && cd ~/libs                # choose a place where you want to build pybullet
git clone -b jazzy https://github.com/SemRoCo/bullet3.git
cd bullet3                                  # be sure to be in the bullet3 folder
./build_better_pybullet.sh                  # this script will also clone and build pybind11 into libs
source ~/.bashrc                            # the script adds a python path modification to your bashrc
```
To test your installation do:
```
workon giskardpy
python3 -c "import betterpybullet"
```
If it doesn't work, make sure that your ```$PYTHONPATH``` includes something like 
```/path/to/your/bullet3/build_cmake/better_python:/path/to/your/bullet3/examples/pybullet```. 

#### Alternative QP solvers
Giskard supports multiple QP solvers and will automatically use the fasted installed solver. 
qpSWIFT is highly recommended.

- `qpalm`: Default solver, because it is the easiest to install and still reasonably fast.
- `qpSWIFT`: Fastest solver for Giskard. Download our fork: [https://github.com/SemRoCo/qpSWIFT](https://github.com/SemRoCo/qpSWIFT.git) and follow the [install instructions](https://github.com/SemRoCo/qpSWIFT/wiki/2.Installation)
- `gurobi`: Commercial solver. Slower than `qpSWIFT`, faster than `qpalm`, useful for debugging during development.
  - ```sudo pip3 install gurobipy```
  - You can apply for a free academic license or buy one here: https://www.gurobi.com/academia/academic-program-and-licenses/
  - If you have vpn access to or are in the local network of the IAI of the University of Bremen, follow these instructions: https://ai.uni-bremen.de/wiki/intern/adm/gurobi

[//]: # (- `Clarabel.rs`: `sudo pip3 install clarabel` &#40;https://github.com/oxfordcontrol/Clarabel.rs&#41;)


### Tutorials
https://github.com/SemRoCo/giskardpy/wiki

### How to cite
Stelter, Simon, Georg Bartels, and Michael Beetz. "An open-source motion planning framework for mobile manipulators using constraint-based task space control with linear MPC." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.

