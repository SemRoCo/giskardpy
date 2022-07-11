#!/usr/bin/env python

import json
import sys
import threading
import time
from collections import namedtuple
from random import uniform
from datetime import datetime

import numpy as np
import rospy
import tf.transformations
import yaml
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from giskard_msgs.msg import Constraint
from nav_msgs.srv import GetMap
from py_trees import Status
import pybullet as p
import giskardpy.model.pybullet_wrapper as pbw

from copy import deepcopy

import giskardpy.identifier as identifier
from giskard_msgs.srv import GlobalPathNeededRequest, GlobalPathNeeded, GetPreGraspRequest, GetPreGrasp, \
    GetAttachedObjects, GetAttachedObjectsRequest
from giskardpy.data_types import PrefixName
from giskardpy.exceptions import GlobalPlanningException, InfeasibleGlobalPlanningException, \
    FeasibleGlobalPlanningException, ReplanningException
from giskardpy.model.pybullet_syncer import PyBulletRayTester, PyBulletMotionValidationIDs, PyBulletBoxSpace
from giskardpy.tree.behaviors.get_goal import GetGoal
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.behaviors.visualization import VisualizationBehavior
from giskardpy.utils.kdl_parser import KDL
from giskardpy.utils.tfwrapper import transform_pose, lookup_pose, np_to_pose_stamped, list_to_kdl, pose_to_np, \
    pose_to_kdl, np_to_pose, pose_stamped_to_np

from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og
from ompl import tools as ot

# todo: put below in ros params
SolveParameters = namedtuple('SolveParameters', 'initial_solve_time refine_solve_time max_initial_iterations '
                                                'max_refine_iterations min_refine_thresh')

from giskardpy.utils.utils import convert_dictionary_to_ros_message, convert_ros_message_to_dictionary, write_to_tmp


class CollisionCheckerInterface(GiskardBehavior):

    def __init__(self, name='CollisionCheckerInterface'):
        super(CollisionCheckerInterface, self).__init__(name)

    def get_collisions(self, link_name):
        all_collisions = self.get_god_map().get_data(identifier.closest_point).items()
        link_collisions = list()
        for c in all_collisions:
            if c.get_link_b() == link_name or c.get_link_a() == link_name:
                link_collisions.append(c)
        return link_collisions


class ObjectGoalOptimizationObjective(ob.PathLengthOptimizationObjective):
    def __init__(self, si, goal, object_name, collision_scene, env_js):
        ob.PathLengthOptimizationObjective.__init__(self, si)
        self.tresh = 0.1
        self.simple_motion_validator = SimpleRayMotionValidator(collision_scene, self.getSpaceInformation(), True, object_name,
                                                                js=env_js)
        self.object_motion_validator = ObjectRayMotionValidator(self.getSpaceInformation(), True, collision_scene,
                                                                object_name, collision_scene.robot, links=[object_name],
                                                                ignore_state_validator=True, js=env_js)
        self.goal = goal()

    def stateCost(self, state):
        return ob.Cost(self.getSpaceInformation().distance(state, self.goal))

    def motionCost(self, s1, s2):
        return self.motionCostHeuristic(s1, s2)

    def motionCostHeuristic(self, s1, s2):

        self.object_motion_validator.raytester.pre_ray_test()
        object_c_free, _, ds, _ = self.object_motion_validator._ompl_check_motion(s1, s2)
        self.object_motion_validator.raytester.post_ray_test()

        self.simple_motion_validator.raytester.pre_ray_test()
        simple_c_free, _, _, _ = self.simple_motion_validator._ompl_check_motion(s1, s2)
        self.simple_motion_validator.raytester.post_ray_test()

        rotation_cost = 0.0
        if simple_c_free and not object_c_free:
            i = 2. * 3.14 / len(ds)
            for d in ds:
                diff = self.tresh - d
                if diff < 0:
                    continue
                else:
                    rotation_cost += (diff / self.tresh) * i
                    break

        return ob.Cost(self.getSpaceInformation().distance(s1, s2) + rotation_cost)


class ReducedOrientationGoalOptimizationObjective(ob.PathLengthOptimizationObjective):
    def __init__(self, si, goal, object_name, collision_scene, env_js):
        ob.PathLengthOptimizationObjective.__init__(self, si)
        self.goal = goal()
        self.percent = 0.9
        self.simple_motion_validator = SimpleRayMotionValidator(collision_scene, self.getSpaceInformation(), True, object_name,
                                                                js=env_js)
        self.object_motion_validator = ObjectRayMotionValidator(self.getSpaceInformation(), True, collision_scene,
                                                                object_name, collision_scene.robot, links=[object_name],
                                                                ignore_state_validator=True, js=env_js)

    # def get_euclidian(self, s1, s2):
    #    return sum((pose_to_np(ompl_se3_state_to_pose(s1))[0] - pose_to_np(ompl_se3_state_to_pose(s2))[0]) ** 2)

    def get_orientation_reduced_cost(self, s1, s2):
        object_c_free = self.object_motion_validator.checkMotion(s1, s2)
        simple_c_free = self.simple_motion_validator.checkMotion(s1, s2)
        rotation_cost = ob.SO3StateSpace().distance(s1.rotation(), s2.rotation())

        if object_c_free and simple_c_free:
            if not self.object_motion_validator.checkMotion(s2, self.goal):
                return ob.SE3StateSpace().distance(s1, s2) - rotation_cost * self.percent

        return ob.SE3StateSpace().distance(s1, s2)

    def motionCost(self, s1, s2):
        return self.motionCostHeuristic(s1, s2)

    def motionCostHeuristic(self, s1, s2):
        return ob.Cost(self.get_orientation_reduced_cost(s1, s2))


def allocPathLengthDirectInfSampler(probDefn, maxNumberCalls):
    return lambda: ob.PathLengthDirectInfSampler(probDefn, maxNumberCalls)


class KindaGoalOptimizationObjective(ob.PathLengthOptimizationObjective):
    def __init__(self, si, h_motion_valid):
        ob.PathLengthOptimizationObjective.__init__(self, si)
        self.h_motion_valid = h_motion_valid

    def setCostToGoHeuristic(self, costToGo):
        self.costToGoFn_ = self.costToGo

    def hasCostToGoHeuristic(self):
        return True

    def costToGo(self, state, goal):
        rospy.logerr('lul')
        if self.h_motion_valid.checkMotion(state, goal):
            return ob.Cost(0.0)
        else:
            return ob.Cost(self.getSpaceInformation().distance(state, goal))


class DynamicSE3GoalSpace(ob.GoalState):

    def __init__(self, si, goal, object_motion_validator, r=0.1, max_samples_in_a_row=5):
        super(DynamicSE3GoalSpace, self).__init__(si)
        self.goal_pose = deepcopy(ompl_se3_state_to_pose(goal))
        self.object_motion_validator = object_motion_validator
        self.max_samples_in_a_row = max_samples_in_a_row
        self.init_goal_space(goal)
        self.init_sampling_space(goal, r)
        self.setState(goal)
        self.setThreshold(0.2)

    def init_goal_space(self, state, diff=0.01):
        self.goal_space = ob.SE3StateSpace()
        bounds = self.goal_space.getBounds()
        bounds.setLow(0, state.getX() - diff)
        bounds.setLow(1, state.getY() - diff)
        bounds.setLow(2, state.getZ() - diff)
        bounds.setHigh(0, state.getX() + diff)
        bounds.setHigh(1, state.getY() + diff)
        bounds.setHigh(2, state.getZ() + diff)
        bounds.check()
        self.goal_space.setBounds(bounds)

    def init_sampling_space(self, state, diff):
        self.sampling_space = ob.SE3StateSpace()
        bounds = self.sampling_space.getBounds()
        bounds.setLow(0, state.getX() - diff)
        bounds.setLow(1, state.getY() - diff)
        bounds.setLow(2, state.getZ() - diff)
        bounds.setHigh(0, state.getX() + diff)
        bounds.setHigh(1, state.getY() + diff)
        bounds.setHigh(2, state.getZ() + diff)
        bounds.check()
        self.sampling_space.setBounds(bounds)
        self.sampler = self.sampling_space.allocStateSampler()

    def sampleGoal(self, state):
        self.sampler.sampleUniform(state)
        if self.object_motion_validator.checkMotionPose(ompl_se3_state_to_pose(state), self.goal_pose):
            self.add_goal_state(state)

    def maxSampleCount(self):
        return self.max_samples_in_a_row

    def aabb_to_2d_points(self, d, u):
        d_l = [d[0], u[1], 0]
        u_r = [u[0], d[1], 0]
        return [d_l, d, u, u_r]

    def aabb_to_3d_points(self, d, u):
        d_new = [d[0], d[1], u[2]]
        u_new = [u[0], u[1], d[2]]
        bottom_cube_face = self.aabb_to_2d_points(d, u_new)
        upper_cube_face = self.aabb_to_2d_points(d_new, u)
        res = []
        for p in bottom_cube_face:
            res.append([p[0], p[1], d[2]])
        for p in upper_cube_face:
            res.append([p[0], p[1], u[2]])
        return res

    def add_goal_state(self, state):
        bounds = self.goal_space.getBounds()
        bounds.setLow(0, min(bounds.low[0], state.getX()))
        bounds.setLow(1, min(bounds.low[1], state.getY()))
        bounds.setLow(2, min(bounds.low[2], state.getZ()))
        bounds.setHigh(0, max(bounds.high[0], state.getX()))
        bounds.setHigh(1, max(bounds.high[1], state.getY()))
        bounds.setHigh(2, max(bounds.high[2], state.getZ()))
        self.goal_space.setBounds(bounds)

    def distanceGoal(self, state):
        if self.goal_space.satisfiesBounds(state):
            return 0.0
        if self.object_motion_validator.checkMotionPose(ompl_se3_state_to_pose(state), self.goal_pose):
            self.add_goal_state(state)
            return 0.0
        else:
            return self.getSpaceInformation().distance(state, self.getState())
            # bounds = self.goal_space.getBounds()
            # d = [bounds.low[0], bounds.low[1], bounds.low[2]]
            # u = [bounds.high[0], bounds.high[1], bounds.high[2]]
            # points = self.aabb_to_3d_points(d, u)
            # min_dist = 10e10
            # state_pose = pose_to_np(ompl_se3_state_to_pose(state))
            # for point in points:
            #    dist = np.sqrt(np.sum((np.array(point) - state_pose[0]) ** 2))
            #    if dist < min_dist:
            #        min_dist = dist
            # return min_dist

    # def isSatisfied(self, *args):
    #    return self.goal_space.satisfiesBounds(args[0])


class GrowingGoalStates(ob.GoalStates):

    def __init__(self, si, robot, root_link, tip_link, start, goal, sampling_axis=None):
        super(GrowingGoalStates, self).__init__(si)
        self.sampling_axis = np.array(sampling_axis, dtype=bool) if sampling_axis is not None else None
        self.start = start
        self.goal = goal
        self.robot = robot
        self.root_link = root_link
        self.tip_link = tip_link
        self.max_consecutive_sampling = 5
        self.max_n_samples = 50
        self.addState(self.goal)

    def maxSampleCount(self):
        return self.getStateCount() \
            if self.getStateCount() < self.max_consecutive_sampling \
            else self.max_consecutive_sampling

    def getGoal(self):
        return self.getState(0)

    def _sampleGoal(self, st):
        for i in range(0, 1):
            # Calc vector from start to goal and roll random rotation around it.
            w_T_gr = self.robot.get_fk(self.root_link, self.tip_link)
            if self.sampling_axis is not None:
                s_arr = np.array([np.random.uniform(low=0.0, high=np.pi)] * 3) * self.sampling_axis
                q_sample = tuple(s_arr.tolist())
            else:
                q_sample = tuple([np.random.uniform(low=0.0, high=np.pi)] * 3)
            gr_q_gr = tf.transformations.quaternion_from_euler(*q_sample)
            w_T_g = tf.transformations.concatenate_matrices(
                tf.transformations.translation_matrix(
                    [self.getGoal().getX(), self.getGoal().getY(), self.getGoal().getZ()]),
                tf.transformations.quaternion_matrix([self.getGoal().rotation().x, self.getGoal().rotation().y,
                                                      self.getGoal().rotation().z, self.getGoal().rotation().w])
            )
            gr_T_goal = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(w_T_gr), w_T_g)
            gr_t_goal = tf.transformations.translation_matrix(tf.transformations.translation_from_matrix(gr_T_goal))
            w_T_calc_g = tf.transformations.concatenate_matrices(
                w_T_gr, tf.transformations.quaternion_matrix(gr_q_gr), gr_t_goal
            )
            q = tf.transformations.quaternion_from_matrix(w_T_calc_g)
            # Apply random rotation around the axis on the goal position and ...
            # state = ob.State(self.getSpaceInformation().getStateSpace())
            st.setX(self.getGoal().getX())
            st.setY(self.getGoal().getY())
            st.setZ(self.getGoal().getZ())
            st.rotation().x = q[0]
            st.rotation().y = q[1]
            st.rotation().z = q[2]
            st.rotation().w = q[3]
            # ... add it to the other goal states, if it is valid.
            if self.getSpaceInformation().isValid(st):
                self.addState(st)

    def sampleGoal(self, st):
        if self.max_n_samples < self.getStateCount():
            super(GrowingGoalStates, self).sampleGoal(st)
        else:
            self._sampleGoal(st)

    def distanceGoal(self, state):
        if self.getSpaceInformation().checkMotion(state, self.getGoal()):
            self.addState(state)
        return super(GrowingGoalStates, self).distanceGoal(state)


class GrowingGoalStatesS(ob.GoalStates):

    def __init__(self, si, robot, root_link, tip_link, start, goal, sampling_axis=None):
        super(GrowingGoalStatesS, self).__init__(si)
        self.sampling_axis = np.array(sampling_axis, dtype=bool) if sampling_axis is not None else None
        self.start = start
        self.goal = goal
        self.robot = robot
        self.root_link = root_link
        self.tip_link = tip_link
        self.addState(self.goal)

    def getGoal(self):
        return self.getState(0)

    def sampleGoal(self, st):
        while True:
            # Calc vector from start to goal and roll random rotation around it.
            w_T_gr = self.robot.get_fk(self.root_link, self.tip_link)
            if self.sampling_axis is not None:
                s_arr = np.array([np.random.uniform(low=0.0, high=np.pi)] * 3) * self.sampling_axis
                q_sample = tuple(s_arr.tolist())
            else:
                q_sample = tuple([np.random.uniform(low=0.0, high=np.pi)] * 3)
            gr_q_gr = tf.transformations.quaternion_from_euler(*q_sample)
            w_T_g = tf.transformations.concatenate_matrices(
                tf.transformations.translation_matrix(
                    [self.getGoal().getX(), self.getGoal().getY(), self.getGoal().getZ()]),
                tf.transformations.quaternion_matrix([self.getGoal().rotation().x, self.getGoal().rotation().y,
                                                      self.getGoal().rotation().z, self.getGoal().rotation().w])
            )
            gr_T_goal = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(w_T_gr), w_T_g)
            gr_t_goal = tf.transformations.translation_matrix(tf.transformations.translation_from_matrix(gr_T_goal))
            w_T_calc_g = tf.transformations.concatenate_matrices(
                w_T_gr, tf.transformations.quaternion_matrix(gr_q_gr), gr_t_goal
            )
            q = tf.transformations.quaternion_from_matrix(w_T_calc_g)
            # Apply random rotation around the axis on the goal position and ...
            # state = ob.State(self.getSpaceInformation().getStateSpace())
            st.setX(self.getGoal().getX())
            st.setY(self.getGoal().getY())
            st.setZ(self.getGoal().getZ())
            st.rotation().x = q[0]
            st.rotation().y = q[1]
            st.rotation().z = q[2]
            st.rotation().w = q[3]
            # ... add it to the other goal states, if it is valid.
            if self.getSpaceInformation().isValid(st):
                self.addState(st)
                break

    #    def distanceGoal(self, state):
    #        if self.getSpaceInformation().checkMotion(state, self.getGoal()):
    #            self.addState(state)
    #        return super(GrowingGoalStatesS, self).distanceGoal(state)

    def isSatisfied(self, *args):
        if len(args) == 1 or len(args) == 2:
            return self.getSpaceInformation().checkMotion(args[0], self.getGoal())
        else:
            raise Exception('Unknown signature.')


class CheckerGoalState(ob.GoalState):

    def __init__(self, si, goal, object_motion_validator):
        super(CheckerGoalState, self).__init__(si)
        self.goal_pose = deepcopy(ompl_se3_state_to_pose(goal))
        self.object_motion_validator = object_motion_validator
        self.setState(goal)
        self.setThreshold(0.0)

    def isSatisfied(self, state):
        return self.object_motion_validator.checkMotionPose(ompl_se3_state_to_pose(state), self.goal_pose)


class CompoundStateSamplerChangingGoalSpace(ob.CompoundStateSampler):

    def __init__(self, sampling_space, goal_state, goal_space, object_motion_validator):
        super(CompoundStateSamplerChangingGoalSpace, self).__init__(sampling_space)
        self.goal = goal_state
        self.goal_space = goal_space
        self.object_motion_validator = object_motion_validator

    def sampleUniform(self, state):
        super(CompoundStateSamplerChangingGoalSpace, self).sampleUniform(state)
        if self.object_motion_validator.checkMotion(state, self.goal):
            self.goal_space.add_goal_state(state)

    def sampleUniformNear(self, state, near, distance):
        super(CompoundStateSamplerChangingGoalSpace, self).sampleUniformNear(state, near, distance)
        if self.object_motion_validator.checkMotion(state, self.goal):
            self.goal_space.add_goal_state(state)

    def sampleGaussian(self, state, mean, stdDev):
        super(CompoundStateSamplerChangingGoalSpace, self).sampleGaussian(state, mean, stdDev)
        if self.object_motion_validator.checkMotion(state, self.goal):
            self.goal_space.add_goal_state(state)


class PathLengthAndGoalOptimizationObjective(ob.PathLengthOptimizationObjective):

    def __init__(self, si, goal):
        ob.PathLengthOptimizationObjective.__init__(self, si)
        self.goal = goal()

    def stateCost(self, state):
        return ob.Cost(self.getSpaceInformation().distance(state, self.goal))

    def motionCost(self, s1, s2):
        return self.motionCostHeuristic(s1, s2)

    def motionCostHeuristic(self, s1, s2):
        return ob.Cost(self.getSpaceInformation().distance(s1, s2))


class OMPLMotionValidator(ob.MotionValidator):
    """
    This class ensures that every Planner in OMPL makes the same assumption for
    planning edges in the resulting path. The state validator can be if needed
    deactivated by passing ignore_state_validator as True. The real motion checking must be implemented
    by overwriting the function ompl_check_motion.
    """

    def __init__(self, si, is_3D, motion_validator):
        ob.MotionValidator.__init__(self, si)
        self.si = si
        self.lock = threading.Lock()
        self.is_3D = is_3D
        self.motion_validator = motion_validator

    def checkMotion(self, *args):
        with self.lock:
            if len(args) == 2:
                s1, s2 = args
            elif len(args) == 3:
                s1, s2, last_valid = args
            else:
                raise Exception('Invalid input arguments.')
            if self.is_3D:
                s1_pose = ompl_se3_state_to_pose(s1)
                s2_pose = ompl_se3_state_to_pose(s2)
            else:
                s1_pose = ompl_se2_state_to_pose(s1)
                s2_pose = ompl_se2_state_to_pose(s2)
            if len(args) == 2:
                return self.motion_validator.checkMotion(s1_pose, s2_pose)
            elif len(args) == 3:
                ret, last_valid_pose, time = self.motion_validator.checkMotionTimed(s1_pose, s2_pose)
                if not ret:
                    valid_state = pose_to_ompl_state(self.si.getStateSpace(), last_valid_pose, self.is_3D)
                    last_valid = (valid_state, time)
                return ret
            else:
                raise Exception('Invalid input arguments.')


class AbstractMotionValidator():
    """
    This class ensures that every Planner in OMPL makes the same assumption for
    planning edges in the resulting path. The state validator can be if needed
    deactivated by passing ignore_state_validator as True. The real motion checking must be implemented
    by overwriting the function ompl_check_motion.
    """

    def __init__(self, tip_link, god_map, ignore_state_validator=False):
        self.god_map = god_map
        self.state_validator = None
        self.tip_link = tip_link
        self.ignore_state_validator = ignore_state_validator

    def get_last_valid(self, s1, s2, f):
        np1 = pose_to_np(s1)
        np2 = pose_to_np(s2)
        np3_trans = np1[0] + (np2[0] - np1[0]) * f
        key_rots = R.from_quat([np1[1], np2[1]])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        times = [f]
        interp_rots = slerp(times)
        np3_rot = interp_rots.as_quat()[0]
        f_trans_m = tf.transformations.translation_matrix(np3_trans)
        f_rot_m = tf.transformations.quaternion_matrix(np3_rot)
        f_m = tf.transformations.concatenate_matrices(f_trans_m, f_rot_m)
        return np_to_pose(f_m)

    def checkMotion(self, s1, s2):
        with self.god_map.get_data(identifier.rosparam + ['motion_validator_lock']):
            res_a = self.check_motion(s1, s2)
            res_b = self.check_motion(s2, s1)
            if self.ignore_state_validator:
                return res_a and res_b
            else:
                return res_a and res_b and \
                       self.state_validator.is_collision_free(s1) and \
                       self.state_validator.is_collision_free(s2)

    def checkMotionTimed(self, s1, s2):
        with self.god_map.get_data(identifier.rosparam + ['motion_validator_lock']):
            c1, f1 = self.check_motion_timed(s1, s2)
            if not self.ignore_state_validator:
                c1 = c1 and self.state_validator.is_collision_free(s1)
            c2, f2 = self.check_motion_timed(s2, s1)
            if not self.ignore_state_validator:
                c2 = c2 and self.state_validator.is_collision_free(s2)
            if not c1:
                time = max(0, f1 - 0.01)
                last_valid = self.get_last_valid(s1, s2, time)
                return False, last_valid, time
            elif not c2:
                # calc_f = 1.0 - f2
                # colliding = True
                # while colliding:
                #    calc_f -= 0.01
                #    colliding, new_f = self.just_check_motion(s1, self.get_last_valid(s1, s2, calc_f))
                # last_valid = self.get_last_valid(s1, s2, max(0, calc_f-0.05))
                return False, s1, 0
            else:
                return True, s2, 1

    def check_motion(self, s1, s2):
        raise Exception('Please overwrite me')

    def check_motion_timed(self, s1, s2):
        raise Exception('Please overwrite me')


# def get_simple_environment_objects_as_urdf(god_map, env_name='kitchen', robot_name):
#     world = god_map.get_data(identifier.world)
#     environment_objects = list()
#     for g_n in world.group_names:
#         if env_name == g_n or robot_name == g_n:
#             continue
#         else:
#             if len(world.groups[g_n].links) == 1:
#                 environment_objects.append(world.groups[g_n].links[g_n].as_urdf())
#     return environment_objects


def get_simple_environment_object_names(god_map, robot_name, env_name='kitchen'):
    world = god_map.get_data(identifier.world)
    environment_objects = list()
    for g_n in world.group_names:
        if env_name == g_n or robot_name == g_n:
            continue
        else:
            if len(world.groups[g_n].links) == 1:
                environment_objects.append(g_n)
    return environment_objects

def current_milli_time():
    return round(time.time() * 1000)


class SimpleRayMotionValidator(AbstractMotionValidator):

    def __init__(self, collision_scene, tip_link, god_map, debug=False, js=None, ignore_state_validator=None):
        AbstractMotionValidator.__init__(self, tip_link, god_map, ignore_state_validator=ignore_state_validator)
        self.hitting = {}
        self.debug = debug
        self.js = deepcopy(js)
        self.debug_times = list()
        self.raytester_lock = threading.Lock()
        environment_object_names = get_simple_environment_object_names(self.god_map)
        self.collision_scene = collision_scene
        self.collision_link_names = self.collision_scene.world.get_children_with_collisions_from_link(self.tip_link)
        pybulletenv = PyBulletMotionValidationIDs(self.god_map.get_data(identifier.collision_scene),
                                                  environment_object_names=environment_object_names,
                                                  moving_links=self.collision_link_names)
        self.raytester = PyBulletRayTester(pybulletenv=pybulletenv)

    def clear(self):
        pass

    def check_motion(self, s1, s2):
        with self.raytester_lock:
            res, _, _, _ = self._ray_test_wrapper(s1, s2)
            return res

    def check_motion_timed(self, s1, s2):
        with self.raytester_lock:
            res, _, _, f = self._ray_test_wrapper(s1, s2)
            return res, f

    def _ray_test_wrapper(self, s1, s2):
        if self.debug:
            s = current_milli_time()
        self.raytester.pre_ray_test()
        collision_free, coll_links, dists, fractions = self._check_motion(s1, s2)
        self.raytester.post_ray_test()
        if self.debug:
            e = current_milli_time()
            self.debug_times.append(e - s)
            rospy.loginfo(f'Motion Validator {self.__class__.__name__}: '
                          f'Raytester: {self.raytester.__class__.__name__}: '
                          f'Summed time: {np.sum(np.array(self.debug_times))} ms.')
        return collision_free, coll_links, dists, fractions

    @profile
    def _check_motion(self, s1, s2):
        # Shoot ray from start to end pose and check if it intersects with the kitchen,
        # if so return false, else true.
        query_b = [[s1.position.x, s1.position.y, s1.position.z]]
        query_e = [[s2.position.x, s2.position.y, s2.position.z]]
        collision_free, coll_links, dists, fractions = self.raytester.ray_test_batch(query_b, query_e)
        return collision_free, coll_links, dists, min(fractions)


class ObjectRayMotionValidator(SimpleRayMotionValidator):

    def __init__(self, collision_scene, tip_link, object_in_motion, state_validator, god_map, debug=False,
                 js=None, ignore_state_validator=False):
        SimpleRayMotionValidator.__init__(self, collision_scene, tip_link, god_map, debug=debug,
                                          js=js, ignore_state_validator=ignore_state_validator)
        self.state_validator = state_validator
        self.object_in_motion = object_in_motion

    @profile
    def _check_motion(self, s1, s2):
        # Shoot ray from start to end pose and check if it intersects with the kitchen,
        # if so return false, else true.
        all_js = self.collision_scene.world.state
        old_js = deepcopy(self.object_in_motion.state)
        state1 = self.state_validator.ik.get_ik(old_js, s1)
        #s = 0.
        #for j_n, v in state1.items():
        #    v2 = self.state_validator.ik.get_ik(old_js, s1)[j_n].position
        #    n = abs(v.position - v2)
        #    if n != 0:
        #        rospy.logerr(f'joint_name: {j_n}: first: {v.position}, second: {v2}, diff: {n}')
        #    s += n
        update_joint_state(all_js, state1)
        self.object_in_motion.state = all_js
        query_b = self.collision_scene.get_aabb_collisions(self.collision_link_names).get_points()
        state2 = self.state_validator.ik.get_ik(old_js, s2)
        #s = 0.
        #for j_n, v in state2.items():
        #    v2 = self.state_validator.ik.get_ik(old_js, s2)[j_n].position
        #    n = abs(v.position - v2)
        #    if n != 0:
        #        rospy.logerr(f'joint_name: {j_n}: first: {v.position}, second: {v2}, diff: {n}')
        #    s += n
        update_joint_state(all_js, state2)
        self.object_in_motion.state = all_js
        query_e = self.collision_scene.get_aabb_collisions(self.collision_link_names).get_points()
        update_joint_state(all_js, old_js)
        self.object_in_motion.state = all_js
        collision_free, coll_links, dists, fractions = self.raytester.ray_test_batch(query_b, query_e)
        return collision_free, coll_links, dists, min(fractions)


class CompoundBoxMotionValidator(AbstractMotionValidator):

    def __init__(self, collision_scene, tip_link, object_in_motion, state_validator, god_map, js=None, links=None):
        super(CompoundBoxMotionValidator, self).__init__(tip_link, god_map)
        self.collision_scene = collision_scene
        self.state_validator = state_validator
        self.object_in_motion = object_in_motion
        environment_object_names = get_simple_environment_object_names(self.god_map, self.god_map.get_data(identifier.robot_group_name))
        self.collision_link_names = self.collision_scene.world.get_children_with_collisions_from_link(self.tip_link)
        pybulletenv = PyBulletMotionValidationIDs(self.god_map.get_data(identifier.collision_scene),
                                                  environment_object_names=environment_object_names,
                                                  moving_links=self.collision_link_names)
        raytester = SimpleRayMotionValidator(self.collision_scene, tip_link, self.god_map,
                                             ignore_state_validator=True,js=js)
        self.box_space = PyBulletBoxSpace(self.collision_scene.world, self.object_in_motion, 'map', pybulletenv)
        # self.collision_points = GiskardPyBulletAABBCollision(object_in_motion, collision_scene, tip_link, links=links)

    def clear(self):
        pass

    @profile
    def check_motion_old(self, s1, s2):
        all_js = self.collision_scene.world.state
        old_js = deepcopy(self.object_in_motion.state)
        ret = True
        for collision_link_name in self.collision_link_names:
            state_ik = self.state_validator.ik.get_ik(old_js, s1)
            update_joint_state(all_js, state_ik)
            self.object_in_motion.state = all_js
            self.collision_scene.sync()
            query_b = pose_stamped_to_np(self.collision_scene.get_pose(collision_link_name))
            state_ik = self.state_validator.ik.get_ik(old_js, s2)
            update_joint_state(all_js, state_ik)
            self.object_in_motion.state = all_js
            self.collision_scene.sync()
            query_e = pose_stamped_to_np(self.collision_scene.get_pose(collision_link_name))
            start_positions = [query_b[0]]
            end_positions = [query_e[0]]
            collision_object = self.collision_scene.get_aabb_info(collision_link_name)
            min_size = np.max(np.abs(np.array(collision_object.d) - np.array(collision_object.u)))
            if self.box_space.is_colliding([min_size], start_positions, end_positions):
                self.object_in_motion.state = all_js
                ret = False
                break
        update_joint_state(all_js, old_js)
        self.object_in_motion.state = all_js
        return ret

    @profile
    def get_box_params(self, s1, s2):
        all_js = self.collision_scene.world.state
        old_js = deepcopy(self.object_in_motion.state)
        state_ik = self.state_validator.ik.get_ik(old_js, s1)
        update_joint_state(all_js, state_ik)
        self.object_in_motion.state = all_js
        self.collision_scene.sync()
        start_positions = list()
        end_positions = list()
        min_sizes = list()
        for collision_link_name in self.collision_link_names:
            start_positions.append(pose_stamped_to_np(self.collision_scene.get_pose(collision_link_name))[0])
        state_ik = self.state_validator.ik.get_ik(old_js, s2)
        update_joint_state(all_js, state_ik)
        self.object_in_motion.state = all_js
        self.collision_scene.sync()
        for collision_link_name in self.collision_link_names:
            end_positions.append(pose_stamped_to_np(self.collision_scene.get_pose(collision_link_name))[0])
            collision_object = self.collision_scene.get_aabb_info(collision_link_name)
            min_size = np.max(np.abs(np.array(collision_object.d) - np.array(collision_object.u)))
            min_sizes.append(min_size)
        update_joint_state(all_js, old_js)
        self.object_in_motion.state = all_js
        return min_sizes, start_positions, end_positions

    @profile
    def check_motion(self, s1, s2):
        return not self.box_space.is_colliding(*self.get_box_params(s1, s2))

    @profile
    def check_motion_timed(self, s1, s2):
        m, s, e = self.get_box_params(s1, s2)
        c, f = self.box_space.is_colliding_timed(m, s, e, pose_to_np(s1)[0], pose_to_np(s2)[0])
        return not c, f


class IK(object):

    def __init__(self, root_link, tip_link):
        self.root_link = root_link
        self.tip_link = tip_link

    def setup(self):
        pass

    def get_ik(self, old_js, pose):
        pass


class KDLIK(IK):

    def __init__(self, root_link, tip_link, static_joints=None, robot_description='robot_description'):
        IK.__init__(self, root_link, tip_link)
        self.robot_description = rospy.get_param(robot_description)
        self._kdl_robot = None
        self.robot_kdl_tree = None
        self.static_joints = static_joints
        self.setup()

    def setup(self):
        self.robot_kdl_tree = KDL(self.robot_description)
        self._kdl_robot = self.robot_kdl_tree.get_robot(self.root_link, self.tip_link, static_joints=self.static_joints)

    def get_ik(self, js, pose):
        new_js = deepcopy(js)
        js_dict_position = {}
        for k, v in js.items():
            js_dict_position[str(k)] = v.position
        joint_array = self._kdl_robot.ik(js_dict_position, pose_to_kdl(pose))
        for i, joint_name in enumerate(self._kdl_robot.joints):
            new_js[PrefixName(joint_name, None)].position = joint_array[i]
        return new_js


class PyBulletIK(IK):

    def __init__(self, root_link, tip_link, static_joints=None, robot_description='robot_description'):
        IK.__init__(self, root_link, tip_link)
        self.robot_description = rospy.get_param(robot_description)
        self.pybullet_joints = list()
        self.robot_id = None
        self.pybullet_tip_link_id = None
        self.once = False
        self.static_joints = static_joints
        self.joint_lowers = list()
        self.joint_uppers = list()
        self.setup()

    def clear(self):
        pbw.p.disconnect(physicsClientId=self.client_id)

    def setup(self):
        for i in range(0, 100):
            if not pbw.p.isConnected(physicsClientId=i):
                self.client_id = i
                break
        pbw.start_pybullet(False, client_id=self.client_id)
        pos, q = pose_to_np(lookup_pose('map', self.root_link).pose)
        self.robot_id = pbw.load_urdf_string_into_bullet(self.robot_description, position=pos,
                                                         orientation=q, client_id=self.client_id)
        for i in range(0, pbw.p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            j = pbw.p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if j[2] != p.JOINT_FIXED:
                joint_name = j[1].decode('UTF-8')
                if self.static_joints and joint_name not in self.static_joints:
                    self.joint_lowers.append(j[8])
                    self.joint_uppers.append(j[9])
                else:
                    self.joint_lowers.append(0)
                    self.joint_uppers.append(0)
                self.pybullet_joints.append(joint_name)
            if j[12].decode('UTF-8') == self.tip_link:
                self.pybullet_tip_link_id = j[0]

    def get_ik(self, js, pose):
        if not self.once:
            self.update_pybullet(js)
            self.once = True
        new_js = deepcopy(js)
        pose = pose_to_np(pose)
        # rospy.logerr(pose[1])
        state_ik = p.calculateInverseKinematics(self.robot_id, self.pybullet_tip_link_id,
                                                pose[0], pose[1], self.joint_lowers, self.joint_uppers,
                                                physicsClientId=self.client_id)
        for joint_name, joint_state in zip(self.pybullet_joints, state_ik):
            new_js[PrefixName(joint_name, None)].position = joint_state
        return new_js

    def update_pybullet(self, js):
        for joint_id in range(0, pbw.p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_name = pbw.p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.client_id)[1].decode()
            joint_state = js[PrefixName(joint_name, None)].position
            pbw.p.resetJointState(self.robot_id, joint_id, joint_state, physicsClientId=self.client_id)
        # pbw.p.stepSimulation(physicsClientId=self.client_id)

    def close_pybullet(self):
        pbw.stop_pybullet(client_id=self.client_id)


class RobotBulletCollisionChecker(GiskardBehavior):

    def __init__(self, is_3D, root_link, tip_link, name='RobotBulletCollisionChecker', ik=None):
        super(RobotBulletCollisionChecker, self).__init__(name)
        self.pybullet_lock = threading.Lock()
        if ik is None:
            self.ik = PyBulletIK(root_link, tip_link)
        else:
            self.ik = ik(root_link, tip_link)
        self.is_3D = is_3D
        self.tip_link = tip_link
        self.setup_pybullet()

    def clear(self):
        self.ik.clear()

    def setup_pybullet(self):
        for i in range(0, 100):
            if not pbw.p.isConnected(physicsClientId=i):
                self.client_id = i
                break
        pbw.start_pybullet(False, client_id=self.client_id)
        self.robot_id = pbw.load_urdf_string_into_bullet(rospy.get_param('robot_description'), client_id=self.client_id)
        self.kitchen_id = pbw.load_urdf_string_into_bullet(rospy.get_param('kitchen_description'),
                                                           client_id=self.client_id)
        self.pybullet_joints_id = dict()
        for i in range(0, p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            j = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if j[2] != p.JOINT_FIXED:
                self.pybullet_joints_id[j[1].decode('UTF-8')] = i

    @profile
    def is_collision_free(self, pose):
        with self.pybullet_lock:
            # Get current joint states
            current_js = self.get_god_map().get_data(identifier.joint_states)
            # Calc IK for navigating to given state and ...
            results = []
            state_ik = self.ik.get_ik(current_js, pose)
            # override on current joint states.
            for joint_name, id in self.pybullet_joints_id.items():
                if joint_name not in ['odom_x_joint', 'odom_y_joint', 'odom_z_joint']:
                    pbw.p.resetJointState(self.robot_id, id, state_ik[joint_name].position,
                                          physicsClientId=self.client_id)
            pbw.p.resetBasePositionAndOrientation(self.robot_id,
                                                  [state_ik['odom_x_joint'].position,
                                                   state_ik['odom_y_joint'].position,
                                                   0],
                                                  p.getQuaternionFromEuler([0, 0, state_ik['odom_z_joint'].position]),
                                                  physicsClientId=self.client_id)
            # pbw.p.stepSimulation(physicsClientId=self.client_id)
            # Check if kitchen is colliding with robot
            for i in range(0, pbw.p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
                aabb = pbw.p.getAABB(self.robot_id, i, physicsClientId=self.client_id)
                aabb = [[v - 0.1 for v in aabb[0]], [v + 0.1 for v in aabb[0]]]
                objs = pbw.p.getOverlappingObjects(aabb[0], aabb[1], physicsClientId=self.client_id)
                results.extend(list(filter(lambda o: o[0] == self.kitchen_id, objs)))
            return len(results) == 0

    def is_collision_free_ompl(self, state):
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        return self.is_collision_free(ompl_state_to_pose(state, self.is_3D))


class GiskardRobotBulletCollisionChecker(GiskardBehavior):

    def __init__(self, is_3D, root_link, tip_link, collision_scene, ik=None, ik_sampling=1, ignore_orientation=False,
                 publish=True, dist=0.0):
        GiskardBehavior.__init__(self, str(self))
        self.giskard_lock = threading.Lock()
        if ik is None:
            self.ik = PyBulletIK(root_link, tip_link)
        else:
            self.ik = ik(root_link, tip_link)
        self.debug = False
        self.debug_times_cc = list()
        self.debug_times = list()
        self.is_3D = is_3D
        self.tip_link = tip_link
        self.dist = dist
        self.ik_sampling = ik_sampling
        self.ignore_orientation = ignore_orientation
        # self.collision_objects = GiskardPyBulletAABBCollision(self.robot, collision_scene, tip_link)
        self.collision_link_names = collision_scene.world.get_children_with_collisions_from_link(self.tip_link)
        self.publisher = None
        if publish:
            self.publisher = VisualizationBehavior('motion planning object publisher', ensure_publish=False)
            self.publisher.setup(10)

    def clear(self):
        self.ik.clear()

    def is_collision_free(self, pose):
        with self.get_god_map().get_data(identifier.rosparam + ['state_validator_lock']):
            # Get current joint states
            all_js = self.collision_scene.world.state
            old_js = deepcopy(self.collision_scene.robot.state)
            # Calc IK for navigating to given state and ...
            results = []
            for i in range(0, self.ik_sampling):
                if self.debug:
                    s_s = current_milli_time()
                state_ik = self.ik.get_ik(deepcopy(self.collision_scene.robot.state), pose)
                # override on current joint states.
                update_joint_state(all_js, state_ik)
                self.robot.state = all_js
                if self.debug:
                    s_c = current_milli_time()
                results.append(
                    self.collision_scene.are_robot_links_external_collision_free(self.collision_link_names,
                                                                                 dist=self.dist))
                if self.debug:
                    e_c = current_milli_time()
                # Reset joint state
                if any(results):
                    self.publish_robot_state()
                update_joint_state(all_js, old_js)
                self.robot.state = all_js
                if self.debug:
                    e_s = current_milli_time()
                    self.debug_times_cc.append(e_c - s_c)
                    self.debug_times.append(e_s - s_s)
                    rospy.loginfo(f'State Validator {self.__class__.__name__}: '
                                  f'CC time: {np.mean(np.array(self.debug_times_cc))} ms. '
                                  f'State Update time: {np.mean(np.array(self.debug_times))-np.mean(np.array(self.debug_times_cc))} ms.')
            return any(results)

    def publish_robot_state(self):
        if self.publisher is not None:
            self.publisher.update()

    def get_furthest_normal(self, pose):
        # Get current joint states
        all_js = self.collision_scene.world.state
        old_js = deepcopy(self.collision_scene.robot.state)
        # Calc IK for navigating to given state and ...
        state_ik = self.ik.get_ik(old_js, pose)
        # override on current joint states.
        update_joint_state(all_js, state_ik)
        self.robot.state = all_js
        # Check if kitchen is colliding with robot
        result = self.collision_scene.get_furthest_normal(self.collision_link_names)
        # Reset joint state
        update_joint_state(all_js, old_js)
        self.robot.state = all_js
        return result

    def get_closest_collision_distance(self, pose, link_names):
        # Get current joint states
        all_js = self.collision_scene.world.state
        old_js = deepcopy(self.collision_scene.robot.state)
        # Calc IK for navigating to given state and ...
        state_ik = self.ik.get_ik(old_js, pose)
        # override on current joint states.
        update_joint_state(all_js, state_ik)
        self.robot.state = all_js
        # Check if kitchen is colliding with robot
        collision = self.collision_scene.get_furthest_collision(link_names)[0]
        # Reset joint state
        update_joint_state(all_js, old_js)
        self.robot.state = all_js
        return collision.contact_distance

    def is_collision_free_ompl(self, state):
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        return self.is_collision_free(ompl_state_to_pose(state, self.is_3D))

    def get_furthest_normal_ompl(self, state):
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        return self.get_furthest_normal(ompl_state_to_pose(state, self.is_3D))

    def get_closest_collision_distance_ompl(self, state, link_names):
        if self.tip_link is None:
            raise Exception(u'Please set tip_link for {}.'.format(str(self)))
        return self.get_closest_collision_distance(ompl_state_to_pose(state, self.is_3D), link_names)


class PyBulletWorldObjectCollisionChecker(GiskardBehavior):

    def __init__(self, is_3D, collision_checker, collision_object_name, collision_offset=0.1):
        GiskardBehavior.__init__(self, str(self))
        self.lock = threading.Lock()
        self.init_pybullet_ids_and_joints()
        self.collision_checker = collision_checker
        self.is_3D = is_3D
        self.collision_object_name = collision_object_name
        self.collision_offset = collision_offset

    def update_object_pose(self, p: Point, q: Quaternion):
        try:
            obj = self.world.groups[self.collision_object_name]
        except KeyError:
            raise Exception(u'Could not find object with name {}.'.format(self.collision_object_name))
        obj.base_pose = Pose(p, q)

    def init_pybullet_ids_and_joints(self):
        if 'pybullet' in sys.modules:
            self.pybullet_initialized = True
        else:
            self.pybullet_initialized = False

    def is_collision_free_ompl(self, state):
        if self.collision_object_name is None:
            raise Exception(u'Please set object for {}.'.format(str(self)))
        if self.pybullet_initialized:
            x = state.getX()
            y = state.getY()
            if self.is_3D:
                z = state.getZ()
                r = state.rotation()
                rot = [r.x, r.y, r.z, r.w]
            else:
                z = 0
                yaw = state.getYaw()
                rot = p.getQuaternionFromEuler([0, 0, yaw])
            # update pose
            self.update_object_pose(Point(x, y, z), Quaternion(rot[0], rot[1], rot[2], rot[3]))
            return self.collision_checker()
        else:
            return True

    def is_collision_free(self, p: Point, q: Quaternion):
        if self.collision_object_name is None:
            raise Exception(u'Please set object for {}.'.format(str(self)))
        if self.pybullet_initialized:
            self.update_object_pose(p, q)
            return self.collision_checker()
        else:
            return True


class ThreeDimStateValidator(ob.StateValidityChecker):

    def __init__(self, si, collision_checker):
        ob.StateValidityChecker.__init__(self, si)
        self.collision_checker = collision_checker
        self.lock = threading.Lock()

    def isValid(self, state):
        with self.lock:
            return self.collision_checker.is_collision_free_ompl(state)

    def isValidPose(self, pose):
        with self.lock:
            return self.collision_checker.is_collision_free(pose)

    def __str__(self):
        return u'ThreeDimStateValidator'


class TwoDimStateValidator(ob.StateValidityChecker):

    def __init__(self, si, collision_checker):
        ob.StateValidityChecker.__init__(self, si)
        self.collision_checker = collision_checker
        self.lock = threading.Lock()
        self.init_map()

    def init_map(self, timeout=3.0):
        try:
            rospy.wait_for_service('static_map', timeout=timeout)
            self.map_initialized = True
        except (rospy.ROSException, rospy.ROSInterruptException) as _:
            rospy.logwarn("Exceeded timeout for map server. Ignoring map...")
            self.map_initialized = False
            return
        try:
            get_map = rospy.ServiceProxy('static_map', GetMap)
            map = get_map().map
            info = map.info
            tmp = numpy.zeros((info.height, info.width))
            for x_i in range(0, info.height):
                for y_i in range(0, info.width):
                    tmp[x_i][y_i] = map.data[y_i + x_i * info.width]
            self.occ_map = numpy.fliplr(deepcopy(tmp))
            self.occ_map_res = info.resolution
            self.occ_map_origin = info.origin.position
            self.occ_map_height = info.height
            self.occ_map_width = info.width
        except rospy.ServiceException as e:
            rospy.logerr("Failed to get static occupancy map. Ignoring map...")
            self.map_initialized = False

    def is_driveable(self, state):
        if self.map_initialized:
            x = numpy.sqrt((state.getX() - self.occ_map_origin.x) ** 2)
            y = numpy.sqrt((state.getY() - self.occ_map_origin.y) ** 2)
            if int(y / self.occ_map_res) >= self.occ_map.shape[0] or \
                    self.occ_map_width - int(x / self.occ_map_res) >= self.occ_map.shape[1]:
                return False
            return 0 <= self.occ_map[int(y / self.occ_map_res)][self.occ_map_width - int(x / self.occ_map_res)] < 1
        else:
            return True

    def isValid(self, state):
        # Some arbitrary condition on the state (note that thanks to
        # dynamic type checking we can just call getX() and do not need
        # to convert state to an SE2State.)
        with self.lock:
            return self.collision_checker.is_collision_free_ompl(state) and self.is_driveable(state)

    def __str__(self):
        return u'TwoDimStateValidator'


def verify_ompl_movement_solution(setup, path, debug=False):
    rbc = setup.getStateValidityChecker().collision_checker
    t = 0
    f = 0
    tj = ompl_states_matrix_to_np(path.printAsMatrix())
    for i in range(0, len(tj)):
        trans = tf.transformations.translation_matrix(np.array([tj[i][0], tj[i][1], tj[i][2]]))
        rot = tf.transformations.quaternion_matrix(np.array([tj[i][3], tj[i][4], tj[i][5], tj[i][6]]))
        pose = np_to_pose(tf.transformations.concatenate_matrices(trans, rot))
        bool = rbc.is_collision_free(pose)
        if bool:
            t += 1
        else:
            f += 1
    if debug:
        rospy.loginfo(u'Num Invalid States: {}, Num Valid States: {}, Rate FP: {}'.format(t, f, f / t))
    return f / t


def verify_ompl_navigation_solution(setup, path, debug=False):
    rbc = setup.getStateValidityChecker().collision_checker
    si = setup.getSpaceInformation()
    t = 0
    f = 0
    tj = ompl_states_matrix_to_np(path.printAsMatrix())
    for i in range(0, len(tj)):
        trans = tf.transformations.translation_matrix(np.array([tj[i][0], tj[i][1], 0]))
        rot = tf.transformations.quaternion_matrix(p.getQuaternionFromEuler([0, 0, tj[i][2]]))
        pose = np_to_pose(tf.transformations.concatenate_matrices(trans, rot))
        bool = rbc.is_collision_free(pose)
        if bool:
            t += 1
        else:
            f += 1
            log_a = u''
            log_b = u''
            c_a = True
            c_b = True
            s_c = ob.State(si.getStateSpace())
            s_c().setX(tj[i][0])
            s_c().setY(tj[i][1])
            s_c().setYaw(tj[i][2])
            if i > 0:
                s_b = ob.State(si.getStateSpace())
                s_b().setX(tj[i - 1][0])
                s_b().setY(tj[i - 1][1])
                s_b().setYaw(tj[i - 1][2])
                c_b_1 = si.checkMotion(s_b(), s_c())
                c_b_2 = si.checkMotion(s_c(), s_b())
                c_b = c_b_1 and c_b_2
                if c_b_1 != c_b_2:
                    rospy.logerr('before to current is not same as current to before:')
                if not c_b:
                    log_b += u' point_b: {}, point_c: {}'.format(tj[i - 1], tj[i])
            if i != len(tj) - 1:
                s_a = ob.State(si.getStateSpace())
                s_a().setX(tj[i + 1][0])
                s_a().setY(tj[i + 1][1])
                s_a().setYaw(tj[i + 1][2])
                c_a_1 = si.checkMotion(s_c(), s_a())
                c_a_2 = si.checkMotion(s_a(), s_c())
                c_a = c_a_1 and c_a_2
                if c_a_1 != c_a_2:
                    rospy.logerr('current to after is not the same as after to current:')
                if not c_a:
                    log_a += ' point_a: {}'.format(tj[i + 1])
            if not (c_a and c_b):
                log = u'c_b: {}, c_a: {}'.format(c_b, c_a) + log_b + log_a
                rospy.logerr(log)
    if debug:
        rospy.loginfo(u'Num Invalid States: {}, Num Valid States: {}, Rate FP: {}'.format(t, f, f / t))
    return f / t


def update_joint_state(js, new_js):
    if any(map(lambda e: type(e) != PrefixName, new_js)):
        raise Exception('oi, there are no PrefixNames in yer new_js >:(!')
    js.update((k, new_js[k]) for k in js.keys() & new_js.keys())


def allocGiskardValidStateSample(si):
    return ob.BridgeTestValidStateSampler(si)


class GiskardValidStateSample(ob.ValidStateSampler):
    def __init__(self, si):
        super(GiskardValidStateSample, self).__init__(si)
        self.name_ = "GiskardValidStateSample"
        self.space = si.getStateSpace()
        self.rbc = GiskardRobotBulletCollisionChecker(False, 'odom_combined', 'base_footprint')

    # Generate a sample in the valid part of the R^3 state space.
    # Valid states satisfy the following constraints:
    # -1<= x,y,z <=1
    # if .25 <= z <= .5, then |x|>.8 and |y|>.8
    def sample(self, state):
        r = ob.State(self.space)
        r.random()
        n = self.rbc.get_furthest_normal_ompl(state)
        state.setX(r().getX() + n[0])
        state.setY(r().getY() + n[1])
        if self.rbc.is_3D:
            state().setZ(r().getZ() + n[2])
        if self.rbc.is_collision_free_ompl(state):
            return True
        else:
            return self.sample(state)


class GoalRegionSampler:
    def __init__(self, is_3D, goal, validation_fun, heuristic_fun, precision=5):
        self.goal = goal
        self.is_3D = is_3D
        self.precision = precision
        self.validation_fun = validation_fun
        self.h = heuristic_fun
        self.valid_samples = list()
        self.samples = list()
        self.goal_list = pose_to_np(goal)[0]

    def _get_pitch(self, pos_b):
        pos_a = np.zeros(3)
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        dz = pos_b[2] - pos_a[2]
        pos_c = pos_a + np.array([dx, dy, 0])
        a = np.sqrt(np.sum((np.array(pos_c) - np.array(pos_a)) ** 2))
        return np.arctan2(dz, a)

    def get_pitch(self, pos_b):
        """ :returns: value in [-np.pi, np.pi] """
        l_a = self._get_pitch(pos_b)
        return -l_a

    def _get_yaw(self, pos_b):
        pos_a = np.zeros(3)
        dx = pos_b[0] - pos_a[0]
        dz = pos_b[2] - pos_a[2]
        pos_c = pos_a + np.array([0, 0, dz])
        pos_d = pos_c + np.array([dx, 0, 0])
        g = np.sqrt(np.sum((np.array(pos_b) - np.array(pos_d)) ** 2))
        a = np.sqrt(np.sum((np.array(pos_c) - np.array(pos_d)) ** 2))
        return np.arctan2(g, a)

    def get_yaw(self, pos_b):
        """ :returns: value in [-2.*np.pi, 2.*np.pi] """
        l_a = self._get_yaw(pos_b)
        if pos_b[0] >= 0 and pos_b[1] >= 0:
            return l_a
        elif pos_b[0] >= 0 and pos_b[1] <= 0:
            return -l_a
        elif pos_b[0] <= 0 and pos_b[1] >= 0:
            return np.pi - l_a
        else:
            return -np.pi + l_a  # todo: -np.pi + l_a

    def valid_sample(self, d=0.5, max_samples=100):
        i = 0
        while i < max_samples:
            valid = False
            while not valid and i < max_samples:
                s = self._sample(i * d / max_samples)
                valid = self.validation_fun(s)
                i += 1
            self.valid_samples.append([self.h(s), s])
        self.valid_samples = sorted(self.valid_samples, key=lambda e: e[0])
        return self.valid_samples[0][1]

    def sample(self, d, samples=100):
        i = 0
        while i < samples:
            s = self._sample(i * d / samples)
            self.samples.append([self.h(s), s])
            i += 1
        self.samples = sorted(self.samples, key=lambda e: e[0])
        return self.samples[0][1]

    def get_orientation(self, p):
        if self.is_3D:
            diff = [self.goal.position.x - p.position.x,
                    self.goal.position.y - p.position.y,
                    self.goal.position.z - p.position.z]
            pitch = self.get_pitch(diff)
            yaw = self.get_yaw(diff)
            return tf.transformations.quaternion_from_euler(0, pitch, yaw)
        else:
            diff = [self.goal.position.x - p.position.x,
                    self.goal.position.y - p.position.y,
                    0]
            yaw = self.get_yaw(diff)
            return tf.transformations.quaternion_from_euler(0, 0, yaw)

    def _sample(self, d):
        s = Pose()
        s.orientation.w = 1

        x = round(uniform(self.goal.position.x - d, self.goal.position.x + d), self.precision)
        y = round(uniform(self.goal.position.y - d, self.goal.position.y + d), self.precision)
        s.position.x = x
        s.position.y = y

        if self.is_3D:
            z = round(uniform(self.goal.position.z - d, self.goal.position.z + d), self.precision)
            s.position.z = z
            q = self.get_orientation(s)
            s.orientation.x = q[0]
            s.orientation.y = q[1]
            s.orientation.z = q[2]
            s.orientation.w = q[3]
        else:
            q = self.get_orientation(s)
            s.orientation.x = q[0]
            s.orientation.y = q[1]
            s.orientation.z = q[2]
            s.orientation.w = q[3]

        return s


class PreGraspSampler(GiskardBehavior):
    def __init__(self, name='PreGraspSampler'):
        super().__init__(name=name)
        self.is_3D = True

    def get_pregrasp_goal(self, cmd):
        try:
            return next(c for c in cmd.constraints if c.type in ['CartesianPreGrasp'])
        except StopIteration:
            return None

    def get_cart_goal(self, cart_c):

        pregrasp_pos = None
        self.__goal_dict = yaml.load(cart_c.parameter_value_pair)
        if 'goal_position' in self.__goal_dict:
            pregrasp_pos = convert_dictionary_to_ros_message(self.__goal_dict[u'goal_position'])
        dist = 0.0
        if 'dist' in self.__goal_dict:
            dist = self.__goal_dict['dist']
        ros_pose = convert_dictionary_to_ros_message(self.__goal_dict[u'grasping_goal'])
        goal = transform_pose(self.get_god_map().get_data(identifier.map_frame), ros_pose)  # type: PoseStamped

        self.root_link = self.__goal_dict[u'root_link']
        tip_link = self.__goal_dict[u'tip_link']
        get_attached_objects = rospy.ServiceProxy('~get_attached_objects', GetAttachedObjects)
        if tip_link in get_attached_objects(GetAttachedObjectsRequest()).object_names:
            tip_link = self.get_robot().get_parent_link_of_link(tip_link)
        self.tip_link = tip_link
        link_names = self.robot.link_names

        if self.root_link not in link_names:
            raise Exception(u'Root_link {} is no known link of the robot.'.format(self.root_link))
        if self.tip_link not in link_names:
            raise Exception(u'Tip_link {} is no known link of the robot.'.format(self.tip_link))
        #if not self.robot.are_linked(self.root_link, self.tip_link):
        #    raise Exception(u'Did not found link chain of the robot from'
        #                    u' root_link {} to tip_link {}.'.format(self.root_link, self.tip_link))

        return goal, pregrasp_pos, dist

    def get_cartesian_pose_constraints(self, goal):

        d = dict()
        d[u'parameter_value_pair'] = deepcopy(self.__goal_dict)

        goal.header.stamp = rospy.Time(0)

        c_d = deepcopy(d)
        c_d[u'parameter_value_pair'][u'goal'] = convert_ros_message_to_dictionary(goal)

        c = Constraint()
        c.type = u'CartesianPreGrasp'
        c.parameter_value_pair = json.dumps(c_d[u'parameter_value_pair'])

        return c

    def update(self):

        # Check if move_cmd exists
        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.SUCCESS

        # Check if move_cmd contains a Cartesian Goal
        cart_c = self.get_pregrasp_goal(move_cmd)
        if not cart_c:
            return Status.SUCCESS

        goal, pregrasp_pos, dist = self.get_cart_goal(cart_c)
        self.collision_scene.update_collision_environment()
        if not self.collision_scene.collision_matrix:
            raise Exception('PreGrasp Sampling is not possible, since the collision matrix is empty')
        if pregrasp_pos is None:
            pregrasp_goal = self.get_pregrasp_cb(goal, self.root_link, self.tip_link, dist=dist)
        else:
            # TODO: check if given position is collision free
            pregrasp_goal = self.get_pregrasp_orientation_cb(pregrasp_pos, goal)

        move_cmd.constraints.remove(cart_c)
        move_cmd.constraints.append(self.get_cartesian_pose_constraints(pregrasp_goal))

        return Status.SUCCESS

    def get_in_map(self, ps):
        return transform_pose('map', ps).pose

    def get_pregrasp_cb(self, goal, root_link, tip_link, dist=0.00):
        collision_scene = self.god_map.unsafe_get_data(identifier.collision_scene)
        robot = collision_scene.robot
        js = self.god_map.unsafe_get_data(identifier.joint_states)
        self.state_validator = GiskardRobotBulletCollisionChecker(self.is_3D, root_link, tip_link,
                                                                  collision_scene, dist=dist)
        self.motion_validator = ObjectRayMotionValidator(collision_scene, tip_link, robot, self.state_validator,
                                                         self.god_map, js=js)

        p, _ = self.sample(js, tip_link, self.get_in_map(goal))
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose = p

        self.state_validator.clear()
        self.motion_validator.clear()

        return ps

    def get_pregrasp_orientation_cb(self, start, goal):
        q_arr = self.get_orientation(self.get_in_map(start), self.get_in_map(goal))
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose = self.get_in_map(start)
        ps.pose.orientation.x = q_arr[0]
        ps.pose.orientation.y = q_arr[1]
        ps.pose.orientation.z = q_arr[2]
        ps.pose.orientation.w = q_arr[3]
        return ps

    def get_orientation(self, curr, goal):
        return GoalRegionSampler(self.is_3D, goal, lambda _: True, lambda _: 0).get_orientation(curr)

    def sample(self, js, tip_link, goal, tries=1000):
        valid_fun = self.state_validator.is_collision_free
        try_i = 0
        next_goals = list()
        while try_i < tries and len(next_goals) == 0:
            goal_grs = GoalRegionSampler(self.is_3D, goal, valid_fun, lambda _: 0)
            s_m = SimpleRayMotionValidator(self.collision_scene, tip_link, self.god_map, ignore_state_validator=True, js=js)
            # Sample goal points which are e.g. in the aabb of the object to pick up
            goal_grs.sample(.5)  # todo d is max aabb size of object
            goals = list()
            next_goals = list()
            for s in goal_grs.samples:
                o_g = s[1]
                if s_m.checkMotion(o_g, goal):
                    goals.append(o_g)
            # Find valid goal poses which allow motion towards the sampled goal points above
            tip_link_grs = GoalRegionSampler(self.is_3D, goal, valid_fun, lambda _: 0)
            motion_cost = lambda a, b: np.sqrt(np.sum((pose_to_np(a)[0] - pose_to_np(b)[0]) ** 2))
            for sampled_goal in goals:
                if len(next_goals) > 0:
                    break
                tip_link_grs.valid_sample(d=.5, max_samples=10)
                for s in tip_link_grs.valid_samples:
                    n_g = s[1]
                    if self.motion_validator.checkMotion(n_g, sampled_goal):
                        next_goals.append([motion_cost(n_g, sampled_goal), n_g, sampled_goal])
                tip_link_grs.valid_samples = list()
            s_m.clear()
            try_i += 1
        if len(next_goals) != 0:
            return sorted(next_goals, key=lambda e: e[0])[0][1], sorted(next_goals, key=lambda e: e[0])[0][2]
        else:
            raise Exception('Could not find PreGrasp samples.')


class GlobalPlanner(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        self.supported_cart_goals = ['CartesianPose', 'CartesianPosition', 'CartesianPathCarrot', 'CartesianPreGrasp']

        # self.robot = self.robot
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.l_tip = 'l_gripper_tool_frame'
        self.r_tip = 'r_gripper_tool_frame'

        self.kitchen_space = self.create_kitchen_space()
        self.kitchen_floor_space = self.create_kitchen_floor_space()

        self.goal = None
        self.__goal_dict = None

        self._planner_solve_params = {}  # todo: load values from rosparam
        self.navigation_config = 'slow_without_refine'  # todo: load value from rosparam
        self.movement_config = 'slow_without_refine'

        self.initialised_planners = False
        self.collisionCheckerInterface = CollisionCheckerInterface()

    def _get_motion_validator_class(self, motion_validator_type, default=None):
        if default is None:
            default = ObjectRayMotionValidator
        if motion_validator_type is not None:
            if motion_validator_type == 'rays':
                return ObjectRayMotionValidator
            elif motion_validator_type == 'box':
                return CompoundBoxMotionValidator
            elif motion_validator_type == 'discrete':
                return None
            else:
                raise Exception('Unknown motion validator class {}.'.format(motion_validator_type))
        else:
            return default

    def get_cart_goal(self, cmd):
        try:
            return next(c for c in cmd.constraints if c.type in self.supported_cart_goals)
        except StopIteration:
            return None

    def is_global_navigation_needed(self):
        return self.tip_link == u'base_footprint' and \
               self.root_link == self.robot.root_link.name

    def save_cart_goal(self, cart_c):

        self.__goal_dict = yaml.load(cart_c.parameter_value_pair)
        ros_pose = convert_dictionary_to_ros_message(self.__goal_dict[u'goal'])
        self.goal = transform_pose(self.get_god_map().get_data(identifier.map_frame), ros_pose)  # type: PoseStamped

        self.root_link = self.__goal_dict[u'root_link']
        tip_link = self.__goal_dict[u'tip_link']
        get_attached_objects = rospy.ServiceProxy('~get_attached_objects', GetAttachedObjects)
        if tip_link in get_attached_objects(GetAttachedObjectsRequest()).object_names:
            tip_link = self.get_robot().get_parent_link_of_link(tip_link)
        self.tip_link = tip_link
        link_names = self.robot.link_names

        if self.root_link not in link_names:
            raise Exception(u'Root_link {} is no known link of the robot.'.format(self.root_link))
        if self.tip_link not in link_names:
            raise Exception(u'Tip_link {} is no known link of the robot.'.format(self.tip_link))
        #if not self.robot.are_linked(self.root_link, self.tip_link):
        #    raise Exception(u'Did not found link chain of the robot from'
        #                    u' root_link {} to tip_link {}.'.format(self.root_link, self.tip_link))

    def seem_trivial(self, simple=True):
        rospy.wait_for_service('~is_global_path_needed', timeout=5.0)
        is_global_path_needed = rospy.ServiceProxy('~is_global_path_needed', GlobalPathNeeded)
        req = GlobalPathNeededRequest()
        req.root_link = self.root_link
        req.tip_link = self.tip_link
        req.env_group = 'kitchen'
        req.pose_goal = self.goal.pose
        req.simple = simple
        return not is_global_path_needed(req).needed

    def _get_navigation_planner(self, planner_name, motion_validator_type, range, time):
        if self.seem_trivial():
            rospy.loginfo('The provided goal might be reached by using CartesianPose,'
                          ' nevertheless continuing with planning for navigating...')
        self.collision_scene.update_collision_environment()
        map_frame = self.get_god_map().get_data(identifier.map_frame)
        motion_validator_class = self._get_motion_validator_class(motion_validator_type)
        state_validator_class = TwoDimStateValidator
        verify_solution_f = verify_ompl_navigation_solution
        dist = 0.1
        planner = MovementPlanner(False, planner_name, state_validator_class, motion_validator_class, range, time,
                                  self.kitchen_floor_space, self.collision_scene,
                                  self.robot, self.root_link, self.tip_link, self.goal.pose, map_frame, self.god_map,
                                  config=self.navigation_config, dist=dist, verify_solution_f=verify_solution_f)
        return planner

    def _get_movement_planner(self, planner_name, motion_validator_type, range, time):
        sampling_goal_axis = self.__goal_dict['goal_sampling_axis'] \
            if 'goal_sampling_axis' in self.__goal_dict \
            else None
        self.collision_scene.update_collision_environment()
        map_frame = self.get_god_map().get_data(identifier.map_frame)
        motion_validator_class = self._get_motion_validator_class(motion_validator_type)
        state_validator_class = ThreeDimStateValidator
        verify_solution_f = verify_ompl_movement_solution
        dist = 0
        planner = MovementPlanner(True, planner_name, state_validator_class, motion_validator_class, range, time,
                                  self.kitchen_space, self.collision_scene,
                                  self.robot, self.root_link, self.tip_link, self.goal.pose, map_frame, self.god_map,
                                  config=self.movement_config, dist=dist, verify_solution_f=verify_solution_f)
        return planner

    def _get_narrow_movement_planner(self, planner_name, motion_validator_type, range, time):
        sampling_goal_axis = self.__goal_dict['goal_sampling_axis'] \
            if 'goal_sampling_axis' in self.__goal_dict \
            else None
        self.collision_scene.update_collision_environment()
        map_frame = self.get_god_map().get_data(identifier.map_frame)
        motion_validator_class = self._get_motion_validator_class(motion_validator_type)
        state_validator_class = ThreeDimStateValidator
        verify_solution_f = verify_ompl_movement_solution
        dist = 0
        planner = NarrowMovementPlanner(planner_name, state_validator_class, motion_validator_class, range, time,
                                        self.kitchen_space, self.collision_scene,
                                        self.robot, self.root_link, self.tip_link, self.goal.pose, map_frame,
                                        self.god_map, config=self.movement_config, dist=dist,
                                        sampling_goal_axis=sampling_goal_axis, verify_solution_f=verify_solution_f)
        return planner

    def get_planner_handle(self, navigation=False, movement=False, narrow=False):
        if not navigation and not movement:
            raise Exception()
        if navigation:
            return self._get_navigation_planner
        elif movement:
            if narrow:
                return self._get_narrow_movement_planner
            else:
                return self._get_movement_planner

    def _plan(self, planner_names, motion_validator_types, planner_range, time,
              navigation=False, movement=False, narrow=False, interpolate=True):
        for motion_validator_type in motion_validator_types:
            for planner_name in planner_names:
                rospy.loginfo(f'Starting planning with Global Planner {planner_name}/{motion_validator_type} ...')
                planner_f = self.get_planner_handle(navigation=navigation, movement=movement, narrow=narrow)
                planner = planner_f(planner_name, motion_validator_type, planner_range, time)
                js = self.get_god_map().get_data(identifier.joint_states)
                try:
                    trajectory = planner.setup_and_plan(js)
                    rospy.logerr(f"Found solution:{len(trajectory) != 0}")
                    if len(trajectory) != 0:
                        if self.god_map.get_data(identifier.path_interpolation):
                            trajectory = planner.interpolate_solution()
                        return trajectory, planner_name, motion_validator_type
                finally:
                    planner.clear()
                rospy.loginfo(f'Global Planner {planner_name}/{motion_validator_type} did not found a solution. '
                              f'Trying other planner config...')
        return None, None, None

    def _benchmark(self, planner_names, motion_validator_types, planner_range, time,
                   navigation=False, movement=False, narrow=False, interpolate=True):
        for motion_validator_type in motion_validator_types:
            rospy.loginfo(f'Starting benchmark with Global Planner {motion_validator_type} ...')
            planner_f = self.get_planner_handle(navigation=navigation, movement=movement, narrow=narrow)
            planner = planner_f(None, motion_validator_type, planner_range, time)
            js = self.get_god_map().get_data(identifier.joint_states)
            try:
                planner.setup_and_benchmark(js, planner_names)
            finally:
                planner.clear()

    def plan(self, navigation=False, movement=False, narrow=False):
        if not navigation and not movement:
            raise Exception()
        if navigation and movement:
            raise Exception()

        if narrow:
            gp_planner_config = self.god_map.get_data(identifier.gp_narrow)
        else:
            gp_planner_config = self.god_map.get_data(identifier.gp_normal)

        if navigation:
            planner_config = gp_planner_config['navigation']
        else:
            planner_config = gp_planner_config['movement']

        motion_validator_types = planner_config['motion_validator']
        planner_names = planner_config['planner']
        planner_range = planner_config['range']
        time = planner_config['time']
        trajectory = None

        if self.god_map.get_data(identifier.path_benchmark):
            self._benchmark(planner_names, motion_validator_types, planner_range, time,
                            navigation=navigation, movement=movement, narrow=narrow)
        else:
            for _ in range(0, int(self.god_map.get_data(identifier.path_replanning_max_retries))):
                try:
                    trajectory, planner_name, _ = self._plan(planner_names, motion_validator_types, planner_range, time,
                                                             navigation=navigation, movement=movement, narrow=narrow)
                    break
                except ReplanningException:
                    pass

        if trajectory is None:
            raise FeasibleGlobalPlanningException('No solution found with current config.')

        if navigation:
            predict_f = 10.0
        else:
            if narrow:
                predict_f = 1.0
            else:
                predict_f = 5.0
        rospy.logerr(predict_f)
        return trajectory, predict_f

    @profile
    def update(self):

        global_planner_needed = self.god_map.get_data(identifier.global_planner_needed)
        if not global_planner_needed:
            return Status.SUCCESS

        # Check if move_cmd exists
        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.SUCCESS

        # Check if move_cmd contains a Cartesian Goal
        cart_c = self.get_cart_goal(move_cmd)
        if not cart_c:
            return Status.SUCCESS

        # Parse and save the Cartesian Goal Constraint
        self.save_cart_goal(cart_c)
        # self.collision_scene.update_collision_environment()
        navigation = global_planner_needed and self.is_global_navigation_needed()
        movement = not navigation
        seem_simple_trivial = self.seem_trivial()
        seem_trivial = self.seem_trivial(simple=False)
        narrow = seem_simple_trivial and not seem_trivial
        try:
            trajectory, predict_f = self.plan(navigation=navigation, movement=movement, narrow=narrow)
        except FeasibleGlobalPlanningException:
            self.raise_to_blackboard(GlobalPlanningException())
            return Status.FAILURE
        poses = []
        for i, point in enumerate(trajectory):
            if i == 0:
                continue  # important assumption for constraint:
                # we do not to reach the first pose, since it is the start pose
            base_pose = PoseStamped()
            base_pose.header.frame_id = self.get_god_map().get_data(identifier.map_frame)
            base_pose.pose.position.x = point[0]
            base_pose.pose.position.y = point[1]
            base_pose.pose.position.z = point[2] if len(point) > 3 else 0
            if len(point) > 3:
                base_pose.pose.orientation = Quaternion(point[3], point[4], point[5], point[6])
            else:
                arr = tf.transformations.quaternion_from_euler(0, 0, point[2])
                base_pose.pose.orientation = Quaternion(arr[0], arr[1], arr[2], arr[3])
            poses.append(base_pose)
        # poses[-1].pose.orientation = self.goal.pose.orientation
        move_cmd.constraints.remove(cart_c)
        move_cmd.constraints.append(self.get_cartesian_path_constraints(poses, predict_f))
        self.get_god_map().set_data(identifier.next_move_goal, move_cmd)
        self.get_god_map().set_data(identifier.global_planner_needed, False)
        return Status.SUCCESS

    def get_cartesian_path_constraints(self, poses, predict_f, ignore_trajectory_orientation=False):

        d = dict()
        d[u'parameter_value_pair'] = deepcopy(self.__goal_dict)

        goal = self.goal
        goal.header.stamp = rospy.Time(0)

        c_d = deepcopy(d)
        c_d[u'parameter_value_pair'][u'goal'] = convert_ros_message_to_dictionary(goal)
        c_d[u'parameter_value_pair'][u'goals'] = list(map(convert_ros_message_to_dictionary, poses))
        c_d[u'parameter_value_pair'][u'predict_f'] = predict_f
        c_d[u'parameter_value_pair'][u'ignore_trajectory_orientation'] = ignore_trajectory_orientation

        c = Constraint()
        c.type = u'CartesianPathCarrot'
        c.parameter_value_pair = json.dumps(c_d[u'parameter_value_pair'])

        return c

    def create_kitchen_space(self):
        # create an SE3 state space
        space = ob.SE3StateSpace()

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(3)

        # Normal
        bounds.setLow(-4)
        bounds.setHigh(4)
        bounds.setLow(2, 0)
        bounds.setHigh(2, 2)

        # Save it
        space.setBounds(bounds)

        # lower distance weight for rotation subspaces
        # for i in range(0, len(space.getSubspaces())):
        #    if 'SO3Space' in space.getSubspace(i).getName():
        #        space.setSubspaceWeight(i, 0.7)
        #    else:
        #        space.setSubspaceWeight(i, 0.3)

        return space

    def create_kitchen_floor_space(self):
        # create an SE2 state space
        space = ob.SE2StateSpace()
        # space.setLongestValidSegmentFraction(0.02)

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-4)
        bounds.setHigh(4)
        space.setBounds(bounds)

        return space


class OMPLPlanner(object):

    def __init__(self, is_3D, planner_name, state_validator_class, motion_validator_class, range, time, space,
                 collision_scene, robot, root_link, tip_link, pose_goal, map_frame, config, god_map,
                 verify_solution_f=None, dist=0.0):
        self.setup = None
        self.is_3D = is_3D
        self.space = space
        self.collision_scene = collision_scene
        self.robot = robot
        self.root_link = root_link
        self.tip_link = tip_link
        self.pose_goal = pose_goal
        self.map_frame = map_frame
        self.config = config
        self.god_map = god_map
        self.planner_name = planner_name
        self.state_validator_class = state_validator_class
        self.motion_validator_class = motion_validator_class
        self.range = range
        self.max_time = time
        self.verify_solution_f = verify_solution_f
        self.dist = dist
        self._planner_solve_params = dict()
        self._planner_solve_params['kABITstar'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5,
                                                   max_initial_iterations=1,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self._planner_solve_params['ABITstar'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5,
                                                   max_initial_iterations=1,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self._planner_solve_params['RRTConnect'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=30, refine_solve_time=5,
                                                   max_initial_iterations=1,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self._planner_solve_params['RRTstar'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=120, refine_solve_time=5,
                                                   max_initial_iterations=1,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self._planner_solve_params['SORRTstar'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=360, refine_solve_time=5,
                                                   max_initial_iterations=1,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self._planner_solve_params['InformedRRTstar'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5,
                                                   max_initial_iterations=1,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }
        self._planner_solve_params[None] = {
            'slow_without_refine': SolveParameters(initial_solve_time=120, refine_solve_time=5,
                                                   max_initial_iterations=1,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'slow_with_refine': SolveParameters(initial_solve_time=60, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5),
            'fast_without_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                   max_refine_iterations=0, min_refine_thresh=0.5),
            'fast_with_refine': SolveParameters(initial_solve_time=15, refine_solve_time=5, max_initial_iterations=3,
                                                max_refine_iterations=5, min_refine_thresh=0.5)
        }

    def clear(self):
        self.setup.clear()

    def get_planner(self, si):
        # rays do not hit the object:
        # RRTConnect w/ 0.05
        # rays hit the object, but orientation needed:
        # InformedRRTstar with ob.Pathlength opt_obja and GrowingStates to terminate
        # EST(+) mit 90% orientation based distance and range of 0.2, no opt_obj
        # STRIDE(+) with range ..075, no opt_obj;
        # BKPIECE1(+) with 90% orientation based and range of 0.2, no opt_obj;
        # setRange SST, LBTRRT, STRIDE/BKPIECE1(+) with rang ..075, no obj; try abit with smaller kitchen space
        # RRTsharp(with opt_obj), SST(with opt_obj), LBTRRT, PDST(many fp), STRIDE, BKPIECE1, FMT
        planner = None
        if self.planner_name is not None:
            # RRT
            if self.planner_name == 'RRT':
                planner = og.RRT(si)
                planner.setRange(self.range)
            elif self.planner_name == 'TRRT':
                planner = og.TRRT(si)
                planner.setRange(self.range)
            elif self.planner_name == 'LazyRRT':
                planner = og.LazyRRT(si)
                planner.setRange(self.range)
            elif self.planner_name == 'RRTConnect':
                planner = og.RRTConnect(si)
                planner.setRange(self.range)
            # elif self.planner_name == 'QRRT':
            #    planner = og.QRRT(si) # todo: fixme
            #    planner.setRange(self.range)
            # RRTstar
            elif self.planner_name == 'RRTstar':
                planner = og.RRTstar(si)
                planner.setRange(self.range)
            elif self.planner_name == 'InformedRRTstar':
                planner = og.InformedRRTstar(si)
                planner.setRange(self.range)
            elif self.planner_name == 'LBTRRT':
                planner = og.LBTRRT(si)
                planner.setRange(self.range)
            elif self.planner_name == 'SST':
                planner = og.SST(si)
                planner.setRange(self.range)
            elif self.planner_name == 'RRTXstatic':
                planner = og.RRTXstatic(si)
                planner.setRange(self.range)
            elif self.planner_name == 'RRTsharp':
                planner = og.RRTsharp(si)
                planner.setRange(self.range)
            # EST
            elif self.planner_name == 'EST':
                planner = og.EST(si)
                planner.setRange(self.range)
            elif self.planner_name == 'SBL':
                planner = og.SBL(si)
                planner.setRange(self.range)
            # KPIECE
            elif self.planner_name == 'KPIECE1':
                planner = og.KPIECE1(si)
                planner.setRange(self.range)
            elif self.planner_name == 'BKPIECE1':
                planner = og.BKPIECE1(si)
                planner.setRange(self.range)
            elif self.planner_name == 'LBKPIECE1':
                planner = og.LBKPIECE1(si)
                planner.setRange(self.range)
            # PRM
            elif self.planner_name == 'PRM':
                planner = og.PRM(si)
            elif self.planner_name == 'LazyPRM':
                planner = og.LazyPRM(si)
            elif self.planner_name == 'PRMstar':
                planner = og.PRMstar(si)
            elif self.planner_name == 'LazyPRMstar':
                planner = og.PRM(si)
            # FMT
            elif self.planner_name == 'FMT':
                planner = og.FMT(si)
            elif self.planner_name == 'BFMT':
                planner = og.BFMT(si)
            # BITstar
            elif self.planner_name == 'BITstar':
                planner = og.BITstar(si)
            elif self.planner_name == 'ABITstar':
                planner = og.ABITstar(si)
            # Other
            elif self.planner_name == 'AITstar':
                planner = og.AITstar(si)
            elif self.planner_name == 'STRIDE':
                planner = og.STRIDE(si)
                planner.setRange(self.range)
            else:
                raise Exception('Planner name {} is not known.'.format(self.planner_name))
        # planner.setSampleRejection(True)
        # planner.setOrderedSampling(True)
        # planner.setInformedSampling(True)
        # planner.setKNearest(False)
        # planner.setKNearest(False)
        # planner.setDelayCC(False)
        # planner = og.ABITstar(si)
        # planner.setIntermediateStates(True)
        # planner.setup()
        return planner

    def init_setup(self):

        # create a simple setup object
        self.setup = og.SimpleSetup(self.space)

        # Set two dimensional motion and state validator
        si = self.setup.getSpaceInformation()
        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(self.allocOBValidStateSampler))
        # si.setStateValidityCheckingResolution(0.001)

        # Set navigation planner
        planner = self.get_planner(si)
        self.setup.setPlanner(planner)

    def get_start_state(self, space, round_decimal_place=3):
        matrix_root_linkTtip_link = self.robot.get_fk(self.root_link, self.tip_link)
        pose_root_linkTtip_link = np_to_pose_stamped(matrix_root_linkTtip_link, self.root_link)
        pose_mTtip_link = transform_pose(self.map_frame, pose_root_linkTtip_link).pose
        return self._get_state(space, pose_mTtip_link, round_decimal_place=round_decimal_place)

    def get_goal_state(self, space, round_decimal_place=3):
        return self._get_state(space, self.pose_goal, round_decimal_place=round_decimal_place)

    def _get_state(self, space, pose_goal, round_decimal_place=3):
        state = ob.State(space)
        state().setX(round(pose_goal.position.x, round_decimal_place))
        state().setY(round(pose_goal.position.y, round_decimal_place))
        o = pose_goal.orientation
        if is_3D(space):
            state().setZ(round(pose_goal.position.z, round_decimal_place))
            state().rotation().x = o.x
            state().rotation().y = o.y
            state().rotation().z = o.z
            state().rotation().w = o.w
        else:
            angles = tf.transformations.euler_from_quaternion([o.x, o.y, o.z, o.w])
            state().setYaw(angles[2])
        # Force goal and start pose into limited search area
        space.enforceBounds(state())
        return state

    @profile
    def next_goal(self, js, goal):
        p = self.god_map.get_data(identifier.tree_manager).get_node('PreGraspSampler')
        collision_scene = self.god_map.get_data(identifier.collision_scene)
        robot = self.god_map.get_data(identifier.robot)
        p.state_validator = GiskardRobotBulletCollisionChecker(self.is_3D, self.root_link, self.tip_link,
                                                               collision_scene, dist=0.0)
        p.motion_validator = ObjectRayMotionValidator(collision_scene, self.tip_link, robot, p.state_validator,
                                                      self.god_map, js=js)
        pose, debug_pose = p.sample(js, self.tip_link, ompl_state_to_pose(goal(), self.is_3D))
        return pose_to_ompl_state(self.space, pose, self.is_3D)

    def shorten_path(self, path, goal):
        if type(goal) == ob.State:
            goal = goal()
        for i in reversed(range(0, path.getStateCount())):
            if not self.setup.getSpaceInformation().checkMotion(path.getState(i), goal):
                including_last_i = i + 2 if i + 2 <= path.getStateCount() else path.getStateCount() - 1
                path.keepBefore(path.getState(including_last_i))

    def _configure_planner(self, planner: ob.Planner):
        try:
            planner.setRange(self.range)
        except AttributeError:
            pass

    def benchmark(self, planner_names):
        e = datetime.now()
        n = f"test_pathAroundKitchenIsland_with_global_planner_and_box-"\
            f"date:={e.day}/{e.month}/{e.year}-" \
            f"time:={e.hour}:{e.minute}:{e.second}-" \
            f"validation type: = {str(self.motion_validator_class)}"
        b = ot.Benchmark(self.setup, n)
        for planner_name in planner_names:
            self.planner_name = planner_name
            b.addPlanner(self.get_planner(self.setup.getSpaceInformation()))
        b.setPreRunEvent(ot.PreSetupEvent(self._configure_planner))
        req = ot.Benchmark.Request()
        req.maxTime = self.max_time
        #req.maxMem = 100.0
        req.runCount = 5
        req.displayProgress = True
        req.simplify = False
        b.benchmark(req)
        b.saveResultsToFile()

    def solve(self):
        # Get solve parameters
        planner_name = self.setup.getPlanner().getName()
        #discrete_checking = self.motion_validator_class is None
        if planner_name not in self._planner_solve_params:
            solve_params = self._planner_solve_params[None][self.config]
        else:
            solve_params = self._planner_solve_params[planner_name][self.config]
        #debugging = sys.gettrace() is not None
        #initial_solve_time = solve_params.initial_solve_time if not debugging else solve_params.initial_solve_time * debug_factor
        #initial_solve_time = initial_solve_time * discrete_factor if discrete_checking else initial_solve_time
        initial_solve_time = self.max_time # min(initial_solve_time, self.max_time)
        #refine_solve_time = solve_params.refine_solve_time if not debugging else solve_params.refine_solve_time * debug_factor
        #refine_solve_time = refine_solve_time * discrete_factor if discrete_checking else refine_solve_time
        max_initial_iterations = solve_params.max_initial_iterations
        #max_refine_iterations = solve_params.max_refine_iterations
        #min_refine_thresh = solve_params.min_refine_thresh
        max_initial_solve_time = min(initial_solve_time * max_initial_iterations, self.max_time)
        #max_refine_solve_time = refine_solve_time * max_refine_iterations

        planner_status = ob.PlannerStatus(ob.PlannerStatus.UNKNOWN)
        num_try = 0
        time_solving_intial = 0
        # Find solution
        while num_try < max_initial_iterations and time_solving_intial < max_initial_solve_time and \
                planner_status.getStatus() not in [ob.PlannerStatus.EXACT_SOLUTION]:
            planner_status = self.setup.solve(initial_solve_time)
            time_solving_intial += self.setup.getLastPlanComputationTime()
            num_try += 1
        # Refine solution
        # refine_iteration = 0
        # v_min = 1e6
        # time_solving_refine = 0
        # if planner_status.getStatus() in [ob.PlannerStatus.EXACT_SOLUTION]:
        #    while v_min > min_refine_thresh and refine_iteration < max_refine_iterations and \
        #            time_solving_refine < max_refine_solve_time:
        #        if 'ABITstar' in self.setup.getPlanner().getName() and min_refine_thresh is not None:
        #            v_before = self.setup.getPlanner().bestCost().value()
        #        self.setup.solve(refine_solve_time)
        #        time_solving_refine += self.setup.getLastPlanComputationTime()
        #        if 'ABITstar' in self.setup.getPlanner().getName() and min_refine_thresh is not None:
        #            v_after = self.setup.getPlanner().bestCost().value()
        #            v_min = v_before - v_after
        #        refine_iteration += 1
        return planner_status.getStatus()

    def get_solution_path(self):
        try:
            path = self.setup.getSolutionPath()
        except RuntimeError:
            raise Exception('Problem Definition in Setup may have changed..')
        return path


class MovementPlanner(OMPLPlanner):

    def __init__(self, is_3D, planner_name, state_validator_class, motion_validator_class, range, time, kitchen_space,
                 collision_scene, robot, root_link, tip_link, pose_goal, map_frame, god_map,
                 config='slow_without_refine', verify_solution_f=None, dist=0.0):
        super(MovementPlanner, self).__init__(is_3D, planner_name, state_validator_class, motion_validator_class, range,
                                              time, kitchen_space, collision_scene, robot, root_link, tip_link,
                                              pose_goal, map_frame, config, god_map,
                                              verify_solution_f=verify_solution_f, dist=dist)

    def clear(self):
        super().clear()
        try:
            self.motion_validator.clear()
        except AttributeError:
            pass
        self.collision_checker.clear()

    def get_planner(self, si):
        # rays do not hit the object:
        # RRTConnect w/ 0.05
        # rays hit the object, but orientation needed:
        # InformedRRTstar with ob.Pathlength opt_obja and GrowingStates to terminate
        # EST(+) mit 90% orientation based distance and range of 0.2, no opt_obj
        # STRIDE(+) with range ..075, no opt_obj;
        # BKPIECE1(+) with 90% orientation based and range of 0.2, no opt_obj;
        # setRange SST, LBTRRT, STRIDE/BKPIECE1(+) with rang ..075, no obj; try abit with smaller kitchen space
        # RRTsharp(with opt_obj), SST(with opt_obj), LBTRRT, PDST(many fp), STRIDE, BKPIECE1, FMT
        planner = super().get_planner(si)
        if planner is None:
            planner = og.RRTConnect(si)
            if self.range is None:
                self.range = 0.05
            planner.setRange(self.range)
            rospy.logwarn('No global planner specified: using RRTConnect with a range of 0.05m.')
        # planner.setSampleRejection(True)
        # planner.setOrderedSampling(True)
        # planner.setInformedSampling(True)
        # planner.setKNearest(False)
        # planner.setKNearest(False)
        # planner.setDelayCC(False)
        # planner = og.ABITstar(si)
        # planner.setIntermediateStates(True)
        # planner.setup()
        return planner

    def setup_problem(self, js):
        self.init_setup()
        si = self.setup.getSpaceInformation()

        # si.getStateSpace().setStateSamplerAllocator()
        self.collision_checker = GiskardRobotBulletCollisionChecker(self.is_3D, self.root_link, self.tip_link,
                                                                    self.collision_scene)
        si.setStateValidityChecker(ThreeDimStateValidator(si, self.collision_checker))

        if self.motion_validator_class is None:
            si.setStateValidityCheckingResolution(1. / ((self.space.getMaximumExtent() * 3) / self.range))
            rospy.loginfo('{}: Using DiscreteMotionValidator with max cost of {} and'
                          ' validity checking resolution of {} where the maximum distance is {}'
                          ' achieving a validity checking distance of {}.'
                          ''.format(str(self.__class__),
                                    self.range,
                                    si.getStateValidityCheckingResolution(),
                                    self.space.getMaximumExtent(),
                                    self.space.getMaximumExtent() * si.getStateValidityCheckingResolution()))
        else:
            # motion_validator = ObjectRayMotionValidator(self.collision_scene, self.tip_link, self.robot, collision_checker, js=js)
            self.motion_validator = self.motion_validator_class(self.collision_scene, self.tip_link, self.robot,
                                                                self.collision_checker, self.god_map, js=js)
            si.setMotionValidator(OMPLMotionValidator(si, self.is_3D, self.motion_validator))
        #si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(allocGiskardValidStateSample))
        si.setup()

        self.start = self.get_start_state(self.space)
        self.goal = self.get_goal_state(self.space)

        if not si.isValid(self.start()):
            rospy.logerr('Start is not valid')
            raise InfeasibleGlobalPlanningException()

        self.setup.setStartAndGoalStates(self.start, self.goal)

        if not si.isValid(self.goal()):
            rospy.logwarn('Goal is not valid, searching new one...')
            goal = self.next_goal(js, self.goal)
            self.setup.setStartAndGoalStates(self.start, goal)

        # h_motion_valid = ObjectRayMotionValidator(si, self.is_3D, self.collision_scene, self.tip_link, self.robot, js=js)
        # optimization_objective = KindaGoalOptimizationObjective(self.setup.getSpaceInformation(), h_motion_valid)
        self.optimization_objective = ob.PathLengthOptimizationObjective(si)
        self.optimization_objective.setCostThreshold(0.1)
        self.setup.setOptimizationObjective(self.optimization_objective)

        self.setup.setup()

    def plan(self):
        return self.solve()

    def setup_and_plan(self, js):
        self.setup_problem(js)
        self._configure_planner(self.setup.getPlanner())
        planner_status = self.plan()
        return self.get_solution(planner_status)

    def setup_and_benchmark(self, js, planner_names):
        self.setup_problem(js)
        self.benchmark(planner_names)

    def interpolate_solution(self):
        path = self.get_solution_path()
        path_cost = path.cost(self.optimization_objective).value()
        if self.is_3D:
            if self.motion_validator_class is None:
                path.interpolate(int(path_cost / 0.01))
            else:
                path.interpolate(int(path_cost / 0.05))
        else:
            path.interpolate(int(path_cost / 0.2))
        data = ompl_states_matrix_to_np(path.printAsMatrix()) # [x y z xw yw zw w]
        if self.verify_solution_f is not None:
            if self.verify_solution_f(self.setup, self.get_solution_path(), debug=True) != 0:
                rospy.loginfo('Interpolated path is invalid. Going to re-plan...')
                raise ReplanningException('Interpolated Path is invalid.')
        else:
            rospy.logwarn('Interpolated path returned is not validated.')
        return data

    def get_solution(self, planner_status, plot=True):
        # og.PathSimplifier(si).smoothBSpline(self.get_solution_path()) # takes around 20-30 secs with RTTConnect(0.1)
        # og.PathSimplifier(si).reduceVertices(self.get_solution_path())
        data = numpy.array([])
        if planner_status in [ob.PlannerStatus.EXACT_SOLUTION]:
            # try to shorten the path
            # self.movement_setup.simplifySolution() DONT! NO! DONT UNCOMMENT THAT! NO! STOP IT! FIRST IMPLEMENT CHECKMOTION! THEN TRY AGAIN!
            # Make sure enough subpaths are available for Path Following
            path = self.get_solution_path()
            data = ompl_states_matrix_to_np(path.printAsMatrix())
            # print the simplified path
            if plot:
                self.plot_solution(data)
        return data

    def plot_solution(self, data, debug=True):
        plt.close()
        dim = '3D' if self.is_3D else '2D'
        if self.verify_solution_f is not None:
            verify = ' - FP: {}, Time: {}s'.format(
                   self.verify_solution_f(self.setup, self.get_solution_path(), debug=debug),
                   self.setup.getLastPlanComputationTime())
        else:
            verify = ''
        path = self.setup.getSolutionPath()
        path_cost = path.cost(self.optimization_objective).value()
        cost = '- Cost: {}'.format(str(round(path_cost, 5)))
        title = u'{} Path from {} in map\n{} {}'.format(dim, self.setup.getPlanner().getName(), verify, cost)
        fig, ax = plt.subplots()
        ax.plot(data[:, 1], data[:, 0])
        ax.invert_xaxis()
        ax.set(xlabel='y (m)', ylabel='x (m)',
               title=title)
        # ax = fig.gca(projection='2d')
        # ax.plot(data[:, 0], data[:, 1], '.-')
        plt.show()


class NarrowMovementPlanner(MovementPlanner):
    def __init__(self, planner_name, state_validator_class, motion_validator_class, range, time, kitchen_space,
                 collision_scene, robot, root_link, tip_link, pose_goal, map_frame, god_map,
                 config='slow_without_refine', dist=0.1, sampling_goal_axis=None, verify_solution_f=None):
        super(NarrowMovementPlanner, self).__init__(True, planner_name, state_validator_class,
                                                    motion_validator_class, range, time, kitchen_space,
                                                    collision_scene, robot, root_link, tip_link,
                                                    pose_goal, map_frame, god_map, config=config, dist=dist,
                                                    verify_solution_f=verify_solution_f)
        self.sampling_goal_axis = sampling_goal_axis
        self.reversed_start_and_goal = False
        self.directional_planner = [
            'RRT', 'TRRT', 'LazyRRT',
            #'EST','KPIECE1', 'BKPIECE1', 'LBKPIECE1', 'FMT',
            #'STRIDE',
            #'BITstar', 'ABITstar', 'kBITstar', 'kABITstar',
            'RRTstar', 'LBTRRT',
            #'SST',
            'RRTXstatic', 'RRTsharp', 'RRT#', 'InformedRRTstar',
            #'SORRTstar'
            ]

    def recompute_start_and_goal(self, planner, start, goal):
        si = self.setup.getSpaceInformation()
        st = ob.State(self.space)
        goal_space_n = 0
        start_space_n = 0
        goal_space = GrowingGoalStates(si, self.robot, self.root_link, self.tip_link, start, goal)
        for i in range(0, 10):
            goal_space.sampleGoal(st())
            if si.isValid(st()):
                goal_space_n += 1
        start_space = GrowingGoalStates(si, self.robot, self.root_link, self.tip_link, goal, start)
        for j in range(0, 10):
            start_space.sampleGoal(st())
            if si.isValid(st()):
                start_space_n += 1
        # Set Goal Space instead of goal state
        prob_def = planner.getProblemDefinition()
        goal_state = prob_def.getGoal().getState()
        if start_space_n > goal_space_n:
            self.reversed_start_and_goal = True
            prob_def.setStartAndGoalStates(goal, start)
            goal_space = GrowingGoalStates(si, self.robot, self.root_link, self.tip_link, goal_state, start(),
                                           sampling_axis=self.sampling_goal_axis)
        else:
            goal_space = GrowingGoalStates(si, self.robot, self.root_link, self.tip_link, start(), goal_state,
                                           sampling_axis=self.sampling_goal_axis)
        goal_space.setThreshold(0.01)
        prob_def.setGoal(goal_space)

    def create_goal_specific_space(self, padding=0.2):

        if self.pose_goal is not None:
            bounds = ob.RealVectorBounds(3)
            curr_p = np_to_pose(self.robot.get_fk(self.root_link, self.tip_link)).position
            s_x = curr_p.x
            s_y = curr_p.y
            s_z = curr_p.z
            goal_p = self.pose_goal.position
            g_x = goal_p.x
            g_y = goal_p.y
            g_z = goal_p.z

            # smaller cereal search space
            bounds.setLow(0, min(s_x, g_x) - padding)
            bounds.setHigh(0, max(s_x, g_x) + padding)
            bounds.setLow(1, min(s_y, g_y) - padding)
            bounds.setHigh(1, max(s_y, g_y) + padding)
            bounds.setLow(2, min(s_z, g_z) - padding)
            bounds.setHigh(2, max(s_z, g_z) + padding)

            # Save it
            self.space.setBounds(bounds)

    def get_solution(self, planner_status, plot=True):
        data = super().get_solution(planner_status, plot=plot)
        if len(data) > 0:
            if not self.reversed_start_and_goal:
                data = np.append(data, [np.append(pose_to_np(self.pose_goal)[0], pose_to_np(self.pose_goal)[1])],
                                 axis=0)
            else:
                data = np.append(data, [np.append(pose_to_np(ompl_se3_state_to_pose(self.start()))[0],
                                                  pose_to_np(ompl_se3_state_to_pose(self.start()))[1])], axis=0)
            data = data if not self.reversed_start_and_goal else np.flip(data, axis=0)
        return data

    def _configure_planner(self, planner: ob.Planner):
        super(NarrowMovementPlanner, self)._configure_planner(planner)
        prob_def = planner.getProblemDefinition()
        try:
            prob_def.getGoal().getState()
        except Exception:
            prob_def.setStartAndGoalStates(self.start, self.goal)
        if planner.getName() in self.directional_planner:
            self.recompute_start_and_goal(planner, self.start, self.goal)

    def setup_problem(self, js):
        super(NarrowMovementPlanner, self).setup_problem(js)
        self.create_goal_specific_space()
        self.setup.setup()



def is_3D(space):
    return type(space) == type(ob.SE3StateSpace())


def ompl_states_matrix_to_np(str: str, line_sep='\n', float_sep=' '):
    states_strings = str.split(line_sep)
    while '' in states_strings:
        states_strings.remove('')
    return numpy.array(list(map(lambda x: numpy.fromstring(x, dtype=float, sep=float_sep), states_strings)))


def ompl_state_to_pose(state, is_3D):
    if is_3D:
        pose = ompl_se3_state_to_pose(state)
    else:
        pose = ompl_se2_state_to_pose(state)
    return pose


def ompl_se3_state_to_pose(state):
    pose = Pose()
    pose.position.x = state.getX()
    pose.position.y = state.getY()
    pose.position.z = state.getZ()
    pose.orientation.x = state.rotation().x
    pose.orientation.y = state.rotation().y
    pose.orientation.z = state.rotation().z
    pose.orientation.w = state.rotation().w
    return pose


def pose_to_ompl_state(space, pose, is_3D):
    if is_3D:
        pose = pose_to_ompl_se3(space, pose)
    else:
        pose = pose_to_ompl_se2(space, pose)
    return pose


def pose_to_ompl_se3(space, pose):
    state = ob.State(space)
    state().setX(pose.position.x)
    state().setY(pose.position.y)
    state().setZ(pose.position.z)
    state().rotation().x = pose.orientation.x
    state().rotation().y = pose.orientation.y
    state().rotation().z = pose.orientation.z
    state().rotation().w = pose.orientation.w
    return state


def pose_to_ompl_se2(space, pose):
    state = ob.State(space)
    state().setX(pose.position.x)
    state().setY(pose.position.y)
    state().setYaw(pose.orientation.z)
    return state


def copy_pose_to_ompl_se3(state, pose):
    state().setX(pose.position.x)
    state().setY(pose.position.y)
    state().setZ(pose.position.z)
    state().rotation().x = pose.orientation.x
    state().rotation().y = pose.orientation.y
    state().rotation().z = pose.orientation.z
    state().rotation().w = pose.orientation.w
    return state


def ompl_se2_state_to_pose(state):
    pose = Pose()
    pose.position.x = state.getX()
    pose.position.y = state.getY()
    pose.position.z = 0
    yaw = state.getYaw()
    rot = tf.transformations.quaternion_from_euler(0, 0, yaw)
    pose.orientation.x = rot[0]
    pose.orientation.y = rot[1]
    pose.orientation.z = rot[2]
    pose.orientation.w = rot[3]
    return pose
