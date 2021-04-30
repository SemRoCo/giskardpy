#!/bin/bash

source ~/ros_ws/devel/setup.bash
cd ~/ros_ws/src/giskardpy
pytest test/test_integration_pr2.py
