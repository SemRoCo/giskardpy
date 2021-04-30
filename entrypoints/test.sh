#!/bin/bash

source ~/ros_ws/devel/setup.bash
cd ~/ros_ws/src/giskardpy
gnome-terminal --command="roslaunch iai_kitchen upload_kitchen_obj.launch"
sleep 5
gnome-terminal --command="roslaunch iai_pr2_sim ros_control_sim_with_base.launch"
sleep 5
pytest test/test_integration_pr2.py
