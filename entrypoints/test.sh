#!/bin/bash

source ~/ros_ws/devel/setup.bash
cd ~/ros_ws/src/giskardpy
roslaunch iai_pr2_sim ros_control_sim_with_base.launch & 
sleep 2 && roslaunch iai_kitchen upload_kitchen_obj.launch & 
pytest test/test_integration_pr2.py
