from __future__ import division

import re
from copy import deepcopy

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped
from numpy import pi
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_matrix, quaternion_about_axis

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import CollisionEntry, MoveResult, WorldBody, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, UpdateWorldRequest
from giskardpy import identifier
from giskardpy.goals.goal import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.python_interface import DEFAULT_WORLD_TIMEOUT
from giskardpy.utils import logging
from utils_for_tests import TestPR2, compare_poses, compare_points, compare_orientations, publish_marker_vector, \
    JointGoalChecker, TestPR2CloseLoop, BoxyCloseLoop


@pytest.fixture(scope='module')
def giskard(request, ros):
    c = BoxyCloseLoop()
    request.addfinalizer(c.tear_down)
    return c


class TestJointGoals(object):
    def test_joint_movement1(self, resetted_giskard):
        """
        :type zero_pose: TestPR2
        """
        resetted_giskard.allow_all_collisions()
        # resetted_giskard.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        resetted_giskard.set_joint_goal(resetted_giskard.default_pose)
        resetted_giskard.plan_and_execute()
        resetted_giskard.allow_all_collisions()
        # resetted_giskard.set_json_goal('SetPredictionHorizon', prediction_horizon=1)
        resetted_giskard.set_joint_goal(resetted_giskard.better_pose)
        resetted_giskard.plan_and_execute()

