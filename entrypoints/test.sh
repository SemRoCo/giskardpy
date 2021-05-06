#!/bin/bash

source ~/ros_ws/devel/setup.bash
cd ~/ros_ws/src/giskardpy
pytest test/test_qpsolver.py
pytest test/test_cas_wrapper.py
