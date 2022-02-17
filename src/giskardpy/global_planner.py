#!/usr/bin/env python

import json
import os
import random
import sys
import threading
from collections import namedtuple
from random import uniform
from time import sleep
import urdf_parser_py.urdf as up

import numpy as np
import rospy
import tf.transformations
import yaml
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import MarkerArray
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from giskard_msgs.msg import Constraint
from nav_msgs.srv import GetMap
from py_trees import Status
import pybullet as p
import giskardpy.model.pybullet_wrapper as pbw

from copy import deepcopy

import giskardpy.identifier as identifier
import giskardpy.model.pybullet_wrapper as pw
from giskard_msgs.srv import GlobalPathNeededRequest, GlobalPathNeeded, GetPreGraspRequest, GetPreGrasp, \
    GetPreGraspOrientation
from giskardpy import RobotName
from giskardpy.exceptions import GlobalPlanningException
from giskardpy.model.utils import make_world_body_box
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.tree.get_goal import GetGoal
from giskardpy.tree.visualization import VisualizationBehavior
from giskardpy.utils.kdl_parser import KDL
from giskardpy.utils.tfwrapper import transform_pose, lookup_pose, np_to_pose_stamped, list_to_kdl, pose_to_np, \
    pose_to_kdl, np_to_pose, pose_stamped_to_np

from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og

# todo: put below in ros params
SolveParameters = namedtuple('SolveParameters', 'initial_solve_time refine_solve_time max_initial_iterations '
                                                'max_refine_iterations min_refine_thresh')
CollisionAABB = namedtuple('CollisionAABB', 'link d u')

from giskardpy.utils.utils import convert_dictionary_to_ros_message, convert_ros_message_to_dictionary, write_to_tmp, \
    resolve_ros_iris_in_urdf


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
        self.simple_motion_validator = SimpleRayMotionValidator(self.getSpaceInformation(), True, object_name,
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
        self.simple_motion_validator = SimpleRayMotionValidator(self.getSpaceInformation(), True, object_name,
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

    def __init__(self, si, robot, root_link, tip_link, start, goal):
        super(GrowingGoalStates, self).__init__(si)
        self.start = start()
        self.goal = goal
        self.robot = robot
        self.root_link = root_link
        self.tip_link = tip_link
        self.addState(self.goal)

    def sampleGoal(self, st):
        for i in range(0, 1):
            # Calc vector from start to goal and roll random rotation around it.
            w_T_gr = self.robot.get_fk(self.root_link, self.tip_link)
            gr_q_gr = tf.transformations.quaternion_from_euler(np.random.uniform(low=0.0, high=np.pi), 0, 0)
            w_T_g = tf.transformations.concatenate_matrices(
                tf.transformations.translation_matrix([self.getState(0).getX(), self.getState(0).getY(), self.getState(0).getZ()]),
                tf.transformations.quaternion_matrix([self.getState(0).rotation().x, self.getState(0).rotation().y,
                                                      self.getState(0).rotation().z, self.getState(0).rotation().w])
            )
            gr_T_goal = tf.transformations.concatenate_matrices(tf.transformations.inverse_matrix(w_T_gr), w_T_g)
            gr_t_goal = tf.transformations.translation_matrix(tf.transformations.translation_from_matrix(gr_T_goal))
            w_T_calc_g = tf.transformations.concatenate_matrices(
                w_T_gr, tf.transformations.quaternion_matrix(gr_q_gr), gr_t_goal
            )
            q = tf.transformations.quaternion_from_matrix(w_T_calc_g)
            # Apply random rotation around the axis on the goal position and ...
            # state = ob.State(self.getSpaceInformation().getStateSpace())
            st.setX(self.getState(0).getX())
            st.setY(self.getState(0).getY())
            st.setZ(self.getState(0).getZ())
            st.rotation().x = q[0]
            st.rotation().y = q[1]
            st.rotation().z = q[2]
            st.rotation().w = q[3]
            # ... add it to the other goal states, if it is valid.
            if self.getSpaceInformation().isValid(st):
                self.addState(st)

    def distanceGoal(self, state):
        if self.getSpaceInformation().checkMotion(state, self.getState(0)):
            self.addState(state)
        return super(GrowingGoalStates, self).distanceGoal(state)


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


class CollisionObjects(object):

    def __init__(self):
        self.collision_objects = list()

    def update(self):
        raise Exception('not implemented')

    def add_collision(self, link_name):
        raise Exception('not implemented')

    def get_collision(self, link_name):
        raise Exception('not implemented')


class AABBCollision(CollisionObjects):

    def __init__(self):
        super(AABBCollision, self).__init__()

    def add_collision(self, link_name):
        link_names = [l for (l, _, _) in self.collision_objects]
        if link_name not in link_names:
            self.collision_objects.append(self.get_collision(link_name))

    def get_points(self, collision_objects=None):
        if collision_objects is None:
            collision_objects = self.collision_objects
        return self._get_points(collision_objects)

    def _get_points(self, collision_objects):
        query = list()
        for collision_info in collision_objects:
            aabbs = self.aabb_to_3d_points(collision_info.d, collision_info.u)
            query.extend(aabbs)
        return query

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

    def aabb_to_2d_points(self, d, u):
        d_l = [d[0], u[1], 0]
        u_r = [u[0], d[1], 0]
        return [d_l, d, u, u_r]


class GiskardLinkCollision(object):

    def __init__(self, object_in_motion, tip_link):
        self.object_in_motion = object_in_motion
        self.tip_link = tip_link

    def get_links(self):
        controlled_parent_joint_name = self.object_in_motion.get_controlled_parent_joint_of_link(self.tip_link)
        child_links_with_collision, _ = self.object_in_motion.search_branch(controlled_parent_joint_name,
                                                                            collect_link_when=self.object_in_motion.has_link_collisions)
        return child_links_with_collision


class GiskardPyBulletAABBCollision(AABBCollision, GiskardLinkCollision):
    # fixme: rework
    # call for motion validator get_points_from_poses twice with s1 and s2. calculate with
    # get_collision the map poses and save them. therefore u can remove transform_points.
    # therefore ros tf is not needed only use pybullet aabb

    def __init__(self, object_in_motion, collision_scene, tip_link, links=None, map_frame='map'):
        AABBCollision.__init__(self)
        GiskardLinkCollision.__init__(self, object_in_motion, tip_link)
        self.map_frame = map_frame
        self.collision_scene = collision_scene
        self.links = links
        self.update()

    def get_links(self):
        if self.links is not None:
            return self.links
        else:
            return super(GiskardPyBulletAABBCollision, self).get_links()

    def update(self):
        """
        If tip link is not used for interaction and has collision information, we use only this information for
        continuous collision checking. Otherwise, we check the parent links of the tip link for fixed links and
        their collision information. If there were no fixed parent links with collision information added,
        we search for collision information in the childs links if possible.
        Further we model interaction tip_links such that it makes sense to search,
        for neighboring links and their collision information. Since the joints
        of these links are normally not fixed, we assume that they are.

        """
        # Add collisions
        self.collision_scene.sync()
        self.collision_objects = list()
        for link_name in self.get_links():
            self.add_collision(link_name)

    def get_points_from_poses(self, collision_objects=None):
        self.update()
        return super(GiskardPyBulletAABBCollision, self).get_points(collision_objects=collision_objects)

    def get_points_from_positions(self):
        self.update()
        return super(GiskardPyBulletAABBCollision, self).get_points(collision_objects=[self.get_cube_collision()])

    @profile
    def get_cube_collision(self, link_names=None, from_link=None):
        d = np.array([1e10] * 3)
        u = np.array([-1e10] * 3)
        if link_names is None:
            link_names = self.get_links()
        if from_link is None:
            from_link = link_names[0]
        for link_name in link_names:
            link_id = self.collision_scene.object_name_to_bullet_id[link_name]
            d_i, u_i = p.getAABB(link_id, physicsClientId=0)
            if (np.array(d_i) < d).all():
                d = d_i
            if (np.array(u_i) > u).all():
                u = u_i
        l = max(abs(np.array(d) - np.array(u)))
        return CollisionAABB(link=from_link, d=tuple(d), u=tuple(d + l))

    def get_distance(self, link_name):
        cc = self.get_cube_collision(link_names=[link_name], from_link=link_name)
        diameter = np.sqrt(sum((np.array(cc.d) - np.array(cc.u)) ** 2))
        c = self.get_collision(link_name)
        l = min(abs(np.array(c.d) - np.array(c.u)))
        return diameter / 2. - l / 2.

    def get_collision(self, link_name):
        if self.object_in_motion.has_link_collisions(link_name):
            link_id = self.collision_scene.object_name_to_bullet_id[link_name]
            aabbs = p.getAABB(link_id, physicsClientId=0)
            return CollisionAABB(link_name, aabbs[0], aabbs[1])


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
                s1, s2, dist = args
            else:
                raise Exception('nope1')
            if self.is_3D:
                s1_pose = ompl_se3_state_to_pose(s1)
                s2_pose = ompl_se3_state_to_pose(s2)
            else:
                s1_pose = ompl_se2_state_to_pose(s1)
                s2_pose = ompl_se2_state_to_pose(s2)
            if len(args) == 2:
                return self.motion_validator.checkMotion(s1_pose, s2_pose)
            elif len(args) == 3:
                ret, last_valid = self.motion_validator.checkMotionTimed(s1_pose, s2_pose, dist)
                return ret, pose_to_ompl_se3(self.si.getStateSpace(), last_valid)
            else:
                raise Exception('nope2')


class AbstractMotionValidator():
    """
    This class ensures that every Planner in OMPL makes the same assumption for
    planning edges in the resulting path. The state validator can be if needed
    deactivated by passing ignore_state_validator as True. The real motion checking must be implemented
    by overwriting the function ompl_check_motion.
    """

    def __init__(self, tip_link, god_map, ignore_state_validator=False, ignore_orientation=False):
        self.god_map = god_map
        self.state_validator = None
        self.tip_link = tip_link
        self.ignore_state_validator = ignore_state_validator
        self.ignore_orientation = ignore_orientation

    def get_last_valid(self, s1, s2, f):
        np1 = pose_to_np(ompl_se3_state_to_pose(s1))
        np2 = pose_to_np(ompl_se3_state_to_pose(s2))
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

    def checkMotionTimed(self, s1, s2, last_valid):
        with self.god_map.get_data(identifier.rosparam + ['motion_validator_lock']):
            c1, f1 = self.check_motion_timed(s1, s2)
            if not self.ignore_state_validator:
                c1 &= self.state_validator.is_collision_free(s1)
            c2, f2 = self.check_motion_timed(s2, s1)
            if not self.ignore_state_validator:
                c2 &= self.state_validator.is_collision_free(s2)
            if not c1:
                last_valid = self.get_last_valid(s1, s2, max(0, f1 - 0.01))
                return False
            elif not c2:
                # calc_f = 1.0 - f2
                # colliding = True
                # while colliding:
                #    calc_f -= 0.01
                #    colliding, new_f = self.just_check_motion(s1, self.get_last_valid(s1, s2, calc_f))
                # last_valid = self.get_last_valid(s1, s2, max(0, calc_f-0.05))
                last_valid = s1
                return False
            else:
                last_valid = s2
                return True

    def check_motion(self, s1, s2):
        raise Exception('Please overwrite me')

    def check_motion_timed(self, s1, s2):
        raise Exception('Please overwrite me')


class PyBulletEnv(object):

    def __init__(self, environment_description='kitchen_description', init_js=None, ignore_objects_ids=None, gui=False):
        self.environment_id = None
        self.environment_description = environment_description
        self.init_js = init_js
        self.gui = gui
        if ignore_objects_ids is None:
            self.ignore_object_ids = list()
        else:
            self.ignore_object_ids = ignore_objects_ids
        self.setup()

    def setup(self):
        for i in range(0, 100):
            if not pbw.p.isConnected(physicsClientId=i):
                self.client_id = i
                break
        pbw.start_pybullet(self.gui, client_id=self.client_id)
        self.environment_id = pbw.load_urdf_string_into_bullet(rospy.get_param(self.environment_description),
                                                               client_id=self.client_id)
        self.update(self.init_js)

    def update(self, js):
        if js is not None:
            # Reset Joint State
            for joint_id in range(0, pbw.p.getNumJoints(self.environment_id, physicsClientId=self.client_id)):
                joint_name = pbw.p.getJointInfo(self.environment_id, joint_id, physicsClientId=self.client_id)[
                    1].decode()
                joint_state = js[joint_name].position
                pbw.p.resetJointState(self.environment_id, joint_id, joint_state, physicsClientId=self.client_id)
            # Recalculate collision stuff
            pbw.p.stepSimulation(
                physicsClientId=self.client_id)  # todo: actually only collision stuff needs to be calculated


class PyBulletRayTester(PyBulletEnv):

    def __init__(self, environment_description='kitchen_description', init_js=None, ignore_objects_ids=None):
        super(PyBulletRayTester, self).__init__(environment_description=environment_description, init_js=init_js, ignore_objects_ids=ignore_objects_ids, gui=False)
        self.once = False
        self.link_id_start = -1
        self.collision_free_id = -1
        self.collisionFilterGroup = 0x1
        self.noCollisionFilterGroup = 0x0

    def pre_ray_test(self):
        bodies_num = p.getNumBodies(physicsClientId=self.client_id)
        if bodies_num > 1:
            for id in range(0, bodies_num):
                links_num = p.getNumJoints(id, physicsClientId=self.client_id)
                for link_id in range(self.link_id_start, links_num):
                    if id not in self.ignore_object_ids:
                        p.setCollisionFilterGroupMask(id, link_id, self.collisionFilterGroup, self.collisionFilterGroup,
                                                      physicsClientId=self.client_id)
                    else:
                        p.setCollisionFilterGroupMask(id, link_id, self.noCollisionFilterGroup,
                                                      self.noCollisionFilterGroup, physicsClientId=self.client_id)

    def ray_test_batch(self, js, rayFromPositions, rayToPositions):
        if not self.once:
            self.update(js)
            self.once = True
        bodies_num = p.getNumBodies(physicsClientId=self.client_id)
        if bodies_num > 1:
            query_res = p.rayTestBatch(rayFromPositions, rayToPositions, numThreads=0, physicsClientId=self.client_id,
                                       collisionFilterMask=self.collisionFilterGroup)
        else:
            query_res = p.rayTestBatch(rayFromPositions, rayToPositions, numThreads=0, physicsClientId=self.client_id)
        if any(v[0] in self.ignore_object_ids for v in query_res):
            rospy.logerr('fak')
        coll_links = []
        dists = []
        for i in range(0, len(query_res)):
            obj_id = query_res[i][0]
            n = query_res[i][-1]
            if obj_id != self.collision_free_id:
                coll_links.append(p.getBodyInfo(obj_id)[0])
                d = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
                dists.append(d)
        return all([v[0] == self.collision_free_id for v in query_res]), coll_links, dists, query_res[0][2]

    def post_ray_test(self):
        bodies_num = p.getNumBodies(physicsClientId=self.client_id)
        if bodies_num > 1:
            for id in range(0, p.getNumBodies(physicsClientId=self.client_id)):
                links_num = p.getNumJoints(id, physicsClientId=self.client_id)
                for link_id in range(self.link_id_start, links_num):
                    p.setCollisionFilterGroupMask(id, link_id, self.collisionFilterGroup, self.collisionFilterGroup,
                                                  physicsClientId=self.client_id)

    def close_pybullet(self):
        pbw.stop_pybullet(client_id=self.client_id)


class SimpleRayMotionValidator(AbstractMotionValidator):

    def __init__(self, tip_link, god_map, debug=False, raytester=None, js=None,
                 ignore_state_validator=None, ignore_orientation=False):
        AbstractMotionValidator.__init__(self, tip_link, god_map,
                                         ignore_state_validator=ignore_state_validator,
                                         ignore_orientation=ignore_orientation)
        self.hitting = {}
        self.debug = debug
        self.js = js
        self.raytester_lock = threading.Lock()
        if raytester is None:
            self.raytester = PyBulletRayTester()
        else:
            self.raytester = raytester

    def check_motion(self, s1, s2):
        with self.raytester_lock:
            self.raytester.pre_ray_test()
            res, _, _, _ = self._check_motion(s1, s2)
            self.raytester.post_ray_test()
            return res

    @profile
    def _check_motion(self, s1, s2):
        # Shoot ray from start to end pose and check if it intersects with the kitchen,
        # if so return false, else true.
        query_b = [[s1.position.x, s1.position.y, s1.position.z]]
        query_e = [[s2.position.x, s2.position.y, s2.position.z]]
        collision_free, coll_links, dists, fraction = self.raytester.ray_test_batch(self.js, query_b, query_e)
        return collision_free, coll_links, dists, fraction


class ObjectRayMotionValidator(SimpleRayMotionValidator):

    def __init__(self, collision_scene, tip_link, object_in_motion, state_validator, god_map, debug=False, raytester=None,
                 js=None, ignore_state_validator=False, links=None, ignore_orientation=False):
        SimpleRayMotionValidator.__init__(self, tip_link, god_map, debug=debug, raytester=raytester,
                                          js=js, ignore_state_validator=ignore_state_validator,
                                          ignore_orientation=ignore_orientation)
        self.state_validator = state_validator
        self.collision_scene = collision_scene
        self.object_in_motion = object_in_motion
        self.collision_points = GiskardPyBulletAABBCollision(object_in_motion, collision_scene, tip_link,
                                                             links=links)

    @profile
    def _check_motion(self, s1, s2):
        # Shoot ray from start to end pose and check if it intersects with the kitchen,
        # if so return false, else true.
        if self.ignore_orientation:
            get_points = self.collision_points.get_points_from_positions
        else:
            get_points = self.collision_points.get_points_from_poses
        all_js = self.collision_scene.world.state
        old_js = self.object_in_motion.state
        state_ik = self.state_validator.ik.get_ik(old_js, s1)
        all_js.update(state_ik)
        self.object_in_motion.state = all_js
        query_b = get_points()
        state_ik = self.state_validator.ik.get_ik(old_js, s2)
        all_js.update(state_ik)
        self.object_in_motion.state = all_js
        query_e = get_points()
        all_js.update(old_js)
        self.object_in_motion.state = all_js
        collision_free, coll_links, dists, fraction = self.raytester.ray_test_batch(self.js, query_b, query_e)
        return collision_free, coll_links, dists, fraction


class TimedObjectRayMotionValidator(ObjectRayMotionValidator):

    def __init__(self, collision_scene, tip_link, object_in_motion, state_validator, god_map, debug=False,
                 raytester=None, js=None, links=None, ignore_state_validator=False, ignore_orientation=False):
        ObjectRayMotionValidator.__init__(self, collision_scene, tip_link, object_in_motion, state_validator,
                                          god_map, debug=debug, raytester=raytester, js=js,
                                          ignore_state_validator=ignore_state_validator,
                                          links=links, ignore_orientation=ignore_orientation)

    def check_motion_timed(self, s1, s2):
        with self.raytester_lock:
            self.raytester.pre_ray_test()
            res, _, _, f = self._check_motion(s1, s2)
            self.raytester.post_ray_test()
            return res, f


class CompoundBoxMotionValidator(AbstractMotionValidator):

    def __init__(self, collision_scene, tip_link, object_in_motion, state_validator, god_map, js=None, links=None):
        super(CompoundBoxMotionValidator, self).__init__(tip_link, god_map)
        self.collision_scene = collision_scene
        self.state_validator = state_validator
        self.object_in_motion = object_in_motion
        self.box_space = CompoundBoxSpace(self.collision_scene.world, self.object_in_motion, 'map', self.collision_scene, js=js)
        self.collision_points = GiskardPyBulletAABBCollision(object_in_motion, collision_scene, tip_link, links=links)

    @profile
    def check_motion(self, s1, s2):
        # Shoot ray from start to end pose and check if it intersects with the kitchen,
        # if so return false, else true.
        all_js = self.collision_scene.world.state
        old_js = self.object_in_motion.state
        for collision_object in self.collision_points.collision_objects:
            state_ik = self.state_validator.ik.get_ik(old_js, s1)
            all_js.update(state_ik)
            self.object_in_motion.state = all_js
            self.collision_scene.sync()
            query_b = pose_stamped_to_np(self.collision_scene.get_pose(collision_object.link))
            state_ik = self.state_validator.ik.get_ik(old_js, s2)
            all_js.update(state_ik)
            self.object_in_motion.state = all_js
            self.collision_scene.sync()
            query_e = pose_stamped_to_np(self.collision_scene.get_pose(collision_object.link))
            start_and_end_positions = [query_b[0], query_e[0]]
            min_size = np.max(np.abs(np.array(collision_object.d) - np.array(collision_object.u)))
            if self.box_space.is_colliding(min_size, start_and_end_positions):
                all_js.update(old_js)
                self.object_in_motion.state = all_js
                return False
        all_js.update(old_js)
        self.object_in_motion.state = all_js
        return True


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
            js_dict_position[k] = v.position
        joint_array = self._kdl_robot.ik(js_dict_position, pose_to_kdl(pose))
        for i, joint_name in enumerate(self._kdl_robot.joints):
            new_js[joint_name].position = joint_array[i]
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
            new_js[joint_name].position = joint_state
        return new_js

    def update_pybullet(self, js):
        for joint_id in range(0, pbw.p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            joint_name = pbw.p.getJointInfo(self.robot_id, joint_id, physicsClientId=self.client_id)[1].decode()
            joint_state = js[joint_name].position
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
        self.is_3D = is_3D
        self.tip_link = tip_link
        self.dist = dist
        self.ik_sampling = ik_sampling
        self.ignore_orientation = ignore_orientation
        self.collision_objects = GiskardPyBulletAABBCollision(self.robot, collision_scene, tip_link)
        self.publisher = None
        if publish:
            self.publisher = VisualizationBehavior('motion planning object publisher', ensure_publish=False)
            self.publisher.setup(10)

    def is_collision_free(self, pose):
        with self.get_god_map().get_data(identifier.rosparam + ['state_validator_lock']):
            # Get current joint states
            all_js = self.collision_scene.world.state
            old_js = self.collision_scene.robot.state
            # Calc IK for navigating to given state and ...
            results = []
            for i in range(0, self.ik_sampling):
                state_ik = self.ik.get_ik(old_js, pose)
                # override on current joint states.
                all_js.update(state_ik)
                self.robot.state = all_js
                # Check if kitchen is colliding with robot
                if self.ignore_orientation:
                    tmp = list()
                    for link_name in self.collision_objects.get_links():
                        dist = self.collision_objects.get_distance(link_name)
                        tmp.append(self.collision_scene.is_robot_link_external_collision_free(link_name, dist=dist))
                    results.append(all(tmp))
                else:
                    results.append(
                        self.collision_scene.are_robot_links_external_collision_free(self.collision_objects.get_links(),
                                                                                     dist=self.dist))
                # Reset joint state
                if any(results):
                    self.publish_robot_state()
                all_js.update(old_js)
                self.robot.state = all_js
            return any(results)

    def publish_robot_state(self):
        if self.publisher is not None:
            self.publisher.update()

    def get_furthest_normal(self, pose):
        # Get current joint states
        all_js = self.collision_scene.world.state
        old_js = self.collision_scene.robot.state
        # Calc IK for navigating to given state and ...
        state_ik = self.ik.get_ik(old_js, pose)
        # override on current joint states.
        all_js.update(state_ik)
        self.robot.state = all_js
        # Check if kitchen is colliding with robot
        result = self.collision_scene.get_furthest_normal(self.collision_link_names)
        # Reset joint state
        all_js.update(old_js)
        self.robot.state = all_js
        return result

    def get_closest_collision_distance(self, pose, link_names):
        # Get current joint states
        all_js = self.collision_scene.world.state
        old_js = self.collision_scene.robot.state
        # Calc IK for navigating to given state and ...
        state_ik = self.ik.get_ik(old_js, pose)
        # override on current joint states.
        all_js.update(state_ik)
        self.robot.state = all_js
        # Check if kitchen is colliding with robot
        collision = self.collision_scene.get_furthest_collision(link_names)[0]
        # Reset joint state
        all_js.update(old_js)
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
            obj = self.get_world().get_object(self.collision_object_name)
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


class CompoundBoxSpace:

    def __init__(self, world, robot, map_frame, collision_scene, publish_collision_boxes=False, js=None):
        self.js = js
        self.pybullet_env = PyBulletEnv(init_js=js)
        self.world = world
        self.robot = robot
        self.map_frame = map_frame
        self.publish_collision_boxes = publish_collision_boxes
        self.collision_scene = collision_scene

        if self.publish_collision_boxes:
            self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)

    def _create_collision_box_with_files(self, pose, pos_a, pos_b, min_size, collision_sphere_name):
        dist = np.sqrt(np.sum((np.array(pos_a) - np.array(pos_b)) ** 2))
        #world_body_box = make_world_body_box(name=collision_sphere_name + 'start',
        #                                     x_length=self.min_size,
        #                                     y_length=self.min_size,
        #                                     z_length=self.min_size)
        #self.world.add_world_body(world_body_box, Pose(Point(pos_a[0], pos_a[1], pos_a[2]), Quaternion(0, 0, 0, 1)))
        #world_body_box = make_world_body_box(name=collision_sphere_name + 'end',
        #                                     x_length=self.min_size,
        #                                     y_length=self.min_size,
        #                                     z_length=self.min_size)
        #self.world.add_world_body(world_body_box, Pose(Point(pos_b[0], pos_b[1], pos_b[2]), Quaternion(0, 0, 0, 1)))
        box_urdf = up.URDF(name=collision_sphere_name)
        box_link = up.Link(name=collision_sphere_name)
        box = up.Box([dist, min_size, min_size])
        box_link.visual = up.Visual(geometry=box)
        box_link.collision = up.Collision(geometry=box)
        box_link.inertial = up.Inertial()
        box_link.inertial.mass = 0.01
        box_link.inertial.inertia = up.Inertia(ixx=0.0, ixy=0.0, ixz=0.0, iyy=0.0, iyz=0.0, izz=0.0)
        box_urdf.add_link(box_link)
        filename = write_to_tmp(u'{}.urdf'.format(pbw.random_string()),  box_urdf.to_xml_string())
        id = p.loadURDF(filename,
                        pose_to_np(pose)[0],
                        pose_to_np(pose)[1],
                        physicsClientId=self.pybullet_env.client_id)
        os.remove(filename)
        return id

    def _create_collision_box(self, pose, pos_a, pos_b, min_size, collision_sphere_name):
        dist = np.sqrt(np.sum((np.array(pos_a) - np.array(pos_b)) ** 2))
        #world_body_box = make_world_body_box(name=collision_sphere_name + 'start',
        #                                     x_length=self.min_size,
        #                                     y_length=self.min_size,
        #                                     z_length=self.min_size)
        #self.world.add_world_body(world_body_box, Pose(Point(pos_a[0], pos_a[1], pos_a[2]), Quaternion(0, 0, 0, 1)))
        #world_body_box = make_world_body_box(name=collision_sphere_name + 'end',
        #                                     x_length=self.min_size,
        #                                     y_length=self.min_size,
        #                                     z_length=self.min_size)
        #self.world.add_world_body(world_body_box, Pose(Point(pos_b[0], pos_b[1], pos_b[2]), Quaternion(0, 0, 0, 1)))
        # the visual shape and collision shape can be re-used by all createMultiBody instances (instancing)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                            halfExtents=[dist/2., min_size/2., min_size/2.],
                                            physicsClientId=self.pybullet_env.client_id)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=[dist/2., min_size/2., min_size/2.],
                                                  physicsClientId=self.pybullet_env.client_id)
        id = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=collisionShapeId,
                               baseVisualShapeIndex=visualShapeId,
                               basePosition=pose_to_np(pose)[0],
                               baseOrientation=pose_to_np(pose)[1],
                               physicsClientId=self.pybullet_env.client_id)
        return id

    def _get_pitch(self, pos_b):
        pos_a = np.zeros(3)
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        dz = pos_b[2] - pos_a[2]
        pos_c = pos_a + np.array([dx, dy, 0])
        a = np.sqrt(np.sum((np.array(pos_c) - np.array(pos_a)) ** 2))
        return np.arctan2(dz, a)

    def get_pitch(self, pos_b):
        l_a = self._get_pitch(pos_b)
        if pos_b[0] >= 0:
            return -l_a
        else:
            return np.pi - l_a

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
        l_a = self._get_yaw(pos_b)
        if pos_b[0] >= 0 and pos_b[1] >= 0:
            return l_a
        elif pos_b[0] >= 0 and pos_b[1] <= 0:
            return -l_a
        elif pos_b[0] <= 0 and pos_b[1] >= 0:
            return np.pi - l_a
        else:
            return np.pi + l_a

    def is_colliding(self, min_size, start_and_end_positions, collision_box_name_prefix=u'is_object_pickable_box'):
        collisions = dict()
        if self.robot:
            collisions_per_pos = dict()
            # Get em
            for i, (pos_a, pos_b) in enumerate(
                    zip(start_and_end_positions[:-1], start_and_end_positions[1:])):
                if np.all(pos_a == pos_b):
                    continue
                b_to_a = np.array(pos_a) - np.array(pos_b)
                c = np.array(pos_b) + b_to_a / 2.
                # https://i.stack.imgur.com/f190Q.png, https://stackoverflow.com/questions/58469297/how-do-i-calculate-the-yaw-pitch-and-roll-of-a-point-in-3d/58469298#58469298
                q = tf.transformations.quaternion_from_euler(0, self.get_pitch(pos_b-c), self.get_yaw(pos_b-c))
                #rospy.logerr(u'pitch: {}, yaw: {}'.format(self._get_pitch(pos_a, pos_b), self._get_yaw(pos_a, pos_b)))
                collision_box_name_i = u'{}_{}'.format(collision_box_name_prefix, str(i))
                # if self.is_tip_object():
                #    tip_coll_aabb = p.getAABB(self.robot.get_pybullet_id(),
                #                              self.robot.get_pybullet_link_id(self.tip_link))
                #    min_size = np.min(np.array(tip_coll_aabb))
                # else:
                #    min_size = box_max_size
                coll_id = self._create_collision_box(Pose(Point(c[0], c[1], c[2]), Quaternion(q[0], q[1], q[2], q[3])),
                                                     pos_a, pos_b, min_size, collision_box_name_i)
                # self.world.set_object_pose(collision_box_name_i, Pose(Point(c[0], c[1], c[2]),
                #                                                      Quaternion(q[0], q[1], q[2], q[3])))
                #if self.publish_collision_boxes:
                #    self.pub_marker(collision_box_name_i)
                self.pybullet_env.update(self.js)
                p.stepSimulation(physicsClientId=self.pybullet_env.client_id)
                contact_points = p.getContactPoints(self.pybullet_env.environment_id, coll_id, physicsClientId=self.pybullet_env.client_id)
                #if self.publish_collision_boxes:
                #    self.del_marker(collision_box_name_i)
                #self.world.remove_object(collision_box_name_i)
                #self.world.remove_object(collision_box_name_i + 'start')
                #self.world.remove_object(collision_box_name_i + 'end')
                p.removeBody(coll_id, physicsClientId=self.pybullet_env.client_id)
                if contact_points != tuple():
                    return True
        return False

    def pub_marker(self, name):
        names = [name, name + 'start', name + 'end']
        ma = MarkerArray()
        for n in names:
            for link_name in self.world.groups[n].link_names:
                markers = self.world.links[link_name].collision_visualization_markers().markers
                for m in markers:
                    m.header.frame_id = self.map_frame
                    m.ns = u'world' + m.ns
                    ma.markers.append(m)
        self.pub_collision_marker.publish(ma)

    def del_marker(self, name):
        names = [name, name + 'start', name + 'end']
        ma = MarkerArray()
        for n in names:
            for link_name in self.world.groups[n].link_names:
                markers = self.world.links[link_name].collision_visualization_markers().markers
                for m in markers:
                    m.action = m.DELETE
                    m.ns = u'world' + m.ns
                    ma.markers.append(m)
        self.pub_collision_marker.publish(ma)


def verify_ompl_movement_solution(setup, debug=False):
    rbc = setup.getStateValidityChecker().collision_checker
    t = 0
    f = 0
    tj = ompl_states_matrix_to_np(setup.getSolutionPath().printAsMatrix())
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


def verify_ompl_navigation_solution(setup, debug=False):
    rbc = setup.getStateValidityChecker().collision_checker
    si = setup.getSpaceInformation()
    t = 0
    f = 0
    tj = ompl_states_matrix_to_np(setup.getSolutionPath().printAsMatrix())
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


def allocGiskardValidStateSample(si):
    return GiskardValidStateSample(si)  # ob.GaussianValidStateSampler(si)


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
        l_a = self._get_pitch(pos_b)
        if pos_b[0] >= 0:
            return -l_a
        else:
            return np.pi - l_a

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
        l_a = self._get_yaw(pos_b)
        if pos_b[0] >= 0 and pos_b[1] >= 0:
            return l_a
        elif pos_b[0] >= 0 and pos_b[1] <= 0:
            return -l_a
        elif pos_b[0] <= 0 and pos_b[1] >= 0:
            return np.pi - l_a
        else:
            return np.pi + l_a # todo: -np.pi + l_a

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
            diff = [self.goal.position.x-p.position.x,
                    self.goal.position.y-p.position.y,
                    self.goal.position.z-p.position.z]
            pitch = self.get_pitch(diff) # fixme: pitch is wrong sometimes
            yaw = self.get_yaw(diff)
            return tf.transformations.quaternion_from_euler(0, 0, yaw)
        else:
            diff = [self.goal.position.x-p.position.x,
                    self.goal.position.y-p.position.y,
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
            q = self.get_orientation(s)
            s.orientation.x = q[0]
            s.orientation.y = q[1]
            s.orientation.z = q[2]
            s.orientation.w = q[3]
            s.position.z = z
        else:
            q = self.get_orientation(s)
            s.orientation.x = q[0]
            s.orientation.y = q[1]
            s.orientation.z = q[2]
            s.orientation.w = q[3]

        return s


class PreGraspSampler(GiskardBehavior):
    def __init__(self, name='PreGraspSampler'):
        super().__init__(name='PreGraspSampler')
        self.is_3D = True

    def setup(self, timeout):
        self.srv_get_pregrasp = rospy.Service(u'~get_pregrasp', GetPreGrasp, self.get_pregrasp_cb)
        self.srv_get_pregrasp_o = rospy.Service(u'~get_pregrasp_orientation', GetPreGraspOrientation,
                                                self.get_pregrasp_orientation_cb)
        return super(PreGraspSampler, self).setup(timeout=timeout)

    def update(self):
        return Status.SUCCESS

    def get_in_map(self, ps):
        return transform_pose('map', ps).pose

    def get_pregrasp_cb(self, req):
        rospy.logerr('0')
        collision_scene = self.god_map.unsafe_get_data(identifier.collision_scene)
        robot = collision_scene.world.groups[RobotName]
        js = self.god_map.unsafe_get_data(identifier.joint_states)
        rospy.logerr('00')
        #collision_scene.sync()
        rospy.logerr('000')
        #collision_scene.sync()
        rospy.logerr(collision_scene.client_id)
        self.state_validator = GiskardRobotBulletCollisionChecker(self.is_3D, req.root_link, req.tip_link,
                                                                  collision_scene, dist=req.dist)
        rospy.logerr('11')
        self.motion_validator = ObjectRayMotionValidator(collision_scene, req.tip_link, robot, self.state_validator,
                                                         self.god_map, js=js)
        rospy.logerr('111')
        p = self.sample(js, req.tip_link, self.get_in_map(req.goal))
        rospy.logerr('1111')
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose = p
        return ps

    def get_pregrasp_orientation_cb(self, req):
        q_arr = self.get_orientation(self.get_in_map(req.start), self.get_in_map(req.goal))
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.pose = self.get_in_map(req.start)
        ps.pose.orientation.x = q_arr[0]
        ps.pose.orientation.y = q_arr[1]
        ps.pose.orientation.z = q_arr[2]
        ps.pose.orientation.w = q_arr[3]
        return ps

    def get_orientation(self, curr, goal):
        return GoalRegionSampler(self.is_3D, goal, lambda _: True, lambda _: 0).get_orientation(curr)

    def sample(self, js, tip_link, goal):
        valid_fun = self.state_validator.is_collision_free
        goal_grs = GoalRegionSampler(self.is_3D, goal, valid_fun, lambda _: 0)
        s_m = SimpleRayMotionValidator(tip_link, self.god_map, ignore_state_validator=True, js=js)
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
        motion_cost = lambda a, b: np.sqrt(np.sum((pose_to_np(a)[0] - pose_to_np(b)[0])**2))
        for sampled_goal in goals:
            if len(next_goals) > 0:
                break
            tip_link_grs.valid_sample(d=.5, max_samples=10)
            for s in tip_link_grs.valid_samples:
                n_g = s[1]
                if self.motion_validator.checkMotion(n_g, sampled_goal):
                    next_goals.append([motion_cost(n_g, sampled_goal), n_g, sampled_goal])
            tip_link_grs.valid_samples = list()
        return sorted(next_goals, key=lambda e: e[0])[0][1], sorted(next_goals, key=lambda e: e[0])[0][2]


class GlobalPlanner(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        self.supported_cart_goals = ['CartesianPose', 'CartesianPosition', 'CartesianPathCarrot']

        # self.robot = self.robot
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.l_tip = 'l_gripper_tool_frame'
        self.r_tip = 'r_gripper_tool_frame'

        self.kitchen_space = self.create_kitchen_space()
        self.kitchen_floor_space = self.create_kitchen_floor_space()

        self.pose_goal = None
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
        self.pose_goal = transform_pose(self.get_god_map().get_data(identifier.map_frame), ros_pose).pose

        self.root_link = self.__goal_dict[u'root_link']
        self.tip_link = self.__goal_dict[u'tip_link']
        link_names = self.robot.link_names

        if self.root_link not in link_names:
            raise Exception(u'Root_link {} is no known link of the robot.'.format(self.root_link))
        if self.tip_link not in link_names:
            raise Exception(u'Tip_link {} is no known link of the robot.'.format(self.tip_link))
        if not self.robot.are_linked(self.root_link, self.tip_link):
            raise Exception(u'Did not found link chain of the robot from'
                            u' root_link {} to tip_link {}.'.format(self.root_link, self.tip_link))

    def seem_trivial(self, simple=True):
        rospy.wait_for_service('~is_global_path_needed', timeout=5.0)
        is_global_path_needed = rospy.ServiceProxy('~is_global_path_needed', GlobalPathNeeded)
        req = GlobalPathNeededRequest()
        req.root_link = self.root_link
        req.tip_link = self.tip_link
        req.env_group = 'kitchen'
        req.pose_goal = self.pose_goal
        req.simple = simple
        return not is_global_path_needed(req).needed

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

        trajectory = np.array([])
        counter = 0
        discrete_checking = False

        while len(trajectory) == 0:
            if global_planner_needed and self.is_global_navigation_needed():
                if self.seem_trivial():
                    rospy.loginfo('The provided goal might be reached by using CartesianPose,'
                                  ' nevertheless continuing with planning...')
                self.collision_scene.update_collision_environment()
                map_frame = self.get_god_map().get_data(identifier.map_frame)
                motion_validator_class = self._get_motion_validator_class(self.get_god_map().get_data(identifier.rosparam + ['global_planner', 'navigation', 'motion_validator']))
                planner = NavigationPlanner(self.kitchen_floor_space, self.collision_scene,
                                            self.robot, self.root_link, self.tip_link, self.pose_goal, map_frame, self.god_map,
                                            config=self.navigation_config, motion_validator_class=motion_validator_class)
                js = self.get_god_map().get_data(identifier.joint_states)
                try:
                    trajectory = planner.plan(js)
                except GlobalPlanningException:
                    self.raise_to_blackboard(GlobalPlanningException())
                    return Status.FAILURE
            elif global_planner_needed:
                seem_simple_trivial = self.seem_trivial()
                seem_trivial = self.seem_trivial(simple=False)
                if seem_trivial:
                    rospy.loginfo('The provided goal might be reached by using CartesianPose,'
                                  ' nevertheless continuing with planning...')
                planner_name = None
                if seem_simple_trivial and not seem_trivial and counter == 0:
                    planner_name = 'InformedRRTstar'
                elif seem_simple_trivial and not seem_trivial and counter == 1:
                    planner_name = 'RTTConnect'
                elif discrete_checking and counter == 2:
                    raise GlobalPlanningException('Global planner did not find a solution.')
                elif counter == 2:
                    discrete_checking = True
                    counter = 0
                    continue
                self.collision_scene.update_collision_environment()
                map_frame = self.get_god_map().get_data(identifier.map_frame)
                motion_validator_class = None
                if not discrete_checking:
                    motion_validator_class = self._get_motion_validator_class(self.get_god_map().get_data(identifier.rosparam + ['global_planner', 'movement', 'motion_validator']))
                planner = MovementPlanner(self.kitchen_space, self.collision_scene,
                                          self.robot, self.root_link, self.tip_link, self.pose_goal, map_frame, self.god_map,
                                          config=self.movement_config, planner_name=planner_name,
                                          discrete_checking=discrete_checking, motion_validator_class=motion_validator_class)
                js = self.get_god_map().get_data(identifier.joint_states)
                try:
                    trajectory = planner.plan(js)
                except GlobalPlanningException:
                    self.raise_to_blackboard(GlobalPlanningException())
                    return Status.FAILURE
            else:
                raise GlobalPlanningException('Global planner was called although it is not needed.')
            counter += 1
            rospy.loginfo('Global Planner did not found a solution. Retrying...')

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
        # poses[-1].pose.orientation = self.pose_goal.pose.orientation
        move_cmd.constraints.remove(cart_c)
        move_cmd.constraints.append(self.get_cartesian_path_constraints(poses))
        self.get_god_map().set_data(identifier.next_move_goal, move_cmd)
        self.get_god_map().set_data(identifier.global_planner_needed, False)
        return Status.SUCCESS

    def get_cartesian_path_constraints(self, poses, ignore_trajectory_orientation=False):

        d = dict()
        d[u'parameter_value_pair'] = deepcopy(self.__goal_dict)

        c_d = deepcopy(d)
        c_d[u'parameter_value_pair'][u'goal'] = self.__goal_dict['goal']
        c_d[u'parameter_value_pair'][u'goals'] = list(map(convert_ros_message_to_dictionary, poses))
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
        bounds.setLow(-10)
        bounds.setHigh(10)
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
        bounds.setLow(-10)
        bounds.setHigh(10)
        space.setBounds(bounds)

        return space


class OMPLPlanner(object):

    def __init__(self, is_3D, space, collision_scene, robot, root_link, tip_link, pose_goal, map_frame, config, god_map,
                 planner_name=None, discrete_checking=False, motion_validator_class=None):
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
        self.discrete_checking = discrete_checking
        self.motion_validator_class = motion_validator_class
        self._planner_solve_params = dict()
        self._planner_solve_params['kABITstar'] = {
            'slow_without_refine': SolveParameters(initial_solve_time=480, refine_solve_time=5,
                                                   max_initial_iterations=3,
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
                                                   max_initial_iterations=3,
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
        self.init_setup()

    def get_planner(self, si):
        raise Exception('Not overwritten.')

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
        p.state_validator = GiskardRobotBulletCollisionChecker(self.is_3D, self.root_link, self.tip_link, collision_scene, dist=0.0)
        p.motion_validator = ObjectRayMotionValidator(collision_scene, self.tip_link, robot, p.state_validator, self.god_map, js=js)
        pose, debug_pose = p.sample(js, self.tip_link, ompl_se3_state_to_pose(goal()))
        return pose_to_ompl_se3(self.space, pose)

    #def get_shorten_path(self, poses):
    #    for i in reversed(range(0, len(poses))):
    #        if not self.motion_validator.check_motion(poses[i].pose, self.pose_goal):
    #            return poses[:i + 2 if i + 2 <= len(poses) else len(poses)]
    #    return poses

    def solve(self, debug_factor=2., discrete_factor=2.):
        # Get solve parameters
        planner_name = self.setup.getPlanner().getName()
        if planner_name not in self._planner_solve_params:
            solve_params = self._planner_solve_params[None][self.config]
        else:
            solve_params = self._planner_solve_params[planner_name][self.config]
        debugging = sys.gettrace() is not None
        initial_solve_time = solve_params.initial_solve_time if not debugging else solve_params.initial_solve_time * debug_factor
        initial_solve_time = initial_solve_time * discrete_factor if self.discrete_checking else initial_solve_time
        refine_solve_time = solve_params.refine_solve_time if not debugging else solve_params.refine_solve_time * debug_factor
        refine_solve_time = refine_solve_time * discrete_factor if self.discrete_checking else refine_solve_time
        max_initial_iterations = solve_params.max_initial_iterations
        max_refine_iterations = solve_params.max_refine_iterations
        min_refine_thresh = solve_params.min_refine_thresh
        max_initial_solve_time = initial_solve_time * max_initial_iterations
        max_refine_solve_time = refine_solve_time * max_refine_iterations

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
        refine_iteration = 0
        v_min = 1e6
        time_solving_refine = 0
        if planner_status.getStatus() in [ob.PlannerStatus.EXACT_SOLUTION]:
            while v_min > min_refine_thresh and refine_iteration < max_refine_iterations and \
                    time_solving_refine < max_refine_solve_time:
                if 'ABITstar' in self.setup.getPlanner().getName() and min_refine_thresh is not None:
                    v_before = self.setup.getPlanner().bestCost().value()
                self.setup.solve(refine_solve_time)
                time_solving_refine += self.setup.getLastPlanComputationTime()
                if 'ABITstar' in self.setup.getPlanner().getName() and min_refine_thresh is not None:
                    v_after = self.setup.getPlanner().bestCost().value()
                    v_min = v_before - v_after
                refine_iteration += 1
        return planner_status.getStatus()


class MovementPlanner(OMPLPlanner):

    def __init__(self, kitchen_space, collision_scene, robot, root_link, tip_link, pose_goal, map_frame, god_map,
                 config='slow_without_refine', planner_name=None, discrete_checking=False, motion_validator_class=None):
        super(MovementPlanner, self).__init__(True, kitchen_space, collision_scene, robot, root_link, tip_link,
                                              pose_goal, map_frame, config, god_map, planner_name=planner_name,
                                              discrete_checking=discrete_checking, motion_validator_class=motion_validator_class)

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
        if self.planner_name is not None:
            if self.planner_name == 'RTTConnect':
                planner = og.RRTConnect(si)
                self.range = 0.05
                planner.setRange(self.range)
            elif self.planner_name == 'InformedRRTstar':
                planner = og.InformedRRTstar(si)
                self.range = 0.05
                planner.setRange(self.range)
            else:
                raise Exception('Planner name {} is not known.'.format(self.planner_name))
        else:
            planner = og.RRTConnect(si)
            self.range = 0.05
            planner.setRange(self.range)
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
            bounds.setLow(0, min(s_x, g_x)-padding)
            bounds.setHigh(0, max(s_x, g_x)+padding)
            bounds.setLow(1, min(s_y, g_y)-padding)
            bounds.setHigh(1, max(s_y, g_y)+padding)
            bounds.setLow(2, min(s_z, g_z)-padding)
            bounds.setHigh(2, max(s_z, g_z)+padding)

            # Save it
            self.space.setBounds(bounds)

    def plan(self, js, plot=True):
        si = self.setup.getSpaceInformation()

        # si.getStateSpace().setStateSamplerAllocator()
        self.collision_checker = GiskardRobotBulletCollisionChecker(self.is_3D, self.root_link, self.tip_link, self.collision_scene)
        si.setStateValidityChecker(ThreeDimStateValidator(si, self.collision_checker))

        if self.discrete_checking:
            si.setStateValidityCheckingResolution(1. / ((self.space.getMaximumExtent() * 3) / self.range))
            rospy.loginfo('MovementPlanner: Using DiscreteMotionValidator with max cost of {} and'
                          ' validity checking resolution of {} where the maximum distance is {}'
                          ' achieving a validity checking distance of {}.'
                          ''.format(self.range,
                                    si.getStateValidityCheckingResolution(),
                                    self.space.getMaximumExtent(),
                                    self.space.getMaximumExtent() * si.getStateValidityCheckingResolution()))
        else:
            #motion_validator = ObjectRayMotionValidator(self.collision_scene, self.tip_link, self.robot, collision_checker, js=js)
            self.motion_validator = self.motion_validator_class(self.collision_scene, self.tip_link, self.robot, self.collision_checker, self.god_map, js=js)
            si.setMotionValidator(OMPLMotionValidator(si, self.is_3D, self.motion_validator))

        if self.setup.getPlanner().getName() == 'InformedRRTstar':
            self.create_goal_specific_space()

        si.setup()

        start = self.get_start_state(self.space)
        goal = self.get_goal_state(self.space)

        if not si.isValid(start()):
            rospy.logerr('start is not valid')
            raise GlobalPlanningException()

        self.setup.setStartAndGoalStates(start, goal)

        if not self.setup.getSpaceInformation().isValid(goal()):
            rospy.logwarn('Goal is not valid, searching new one...')
            next_goal = self.next_goal(js, goal)
            self.setup.setStartAndGoalStates(start, next_goal)

        if self.setup.getPlanner().getName() == 'InformedRRTstar':
        # Set Goal Space instead of goal state
            goal_state = self.setup.getGoal().getState()
            goal_space = GrowingGoalStates(si, self.robot, self.root_link, self.tip_link, start, goal_state)
            goal_space.setThreshold(0.01)
            self.setup.setGoal(goal_space)

        # h_motion_valid = ObjectRayMotionValidator(si, self.is_3D, self.collision_scene, self.tip_link, self.robot, js=js)
        # optimization_objective = KindaGoalOptimizationObjective(self.setup.getSpaceInformation(), h_motion_valid)
        optimization_objective = ob.PathLengthOptimizationObjective(si)
        optimization_objective.setCostThreshold(0.1)
        self.setup.setOptimizationObjective(optimization_objective)

        self.setup.setup()
        planner_status = self.solve()
        # og.PathSimplifier(si).smoothBSpline(ss.getSolutionPath()) # takes around 20-30 secs with RTTConnect(0.1)
        # og.PathSimplifier(si).reduceVertices(ss.getSolutionPath())
        data = numpy.array([])
        if planner_status in [ob.PlannerStatus.EXACT_SOLUTION]:
            # try to shorten the path
            # self.movement_setup.simplifySolution() DONT! NO! DONT UNCOMMENT THAT! NO! STOP IT! FIRST IMPLEMENT CHECKMOTION! THEN TRY AGAIN!
            # Make sure enough subpaths are available for Path Following
            self.setup.getSolutionPath().interpolate(20)

            data = ompl_states_matrix_to_np(self.setup.getSolutionPath().printAsMatrix())  # [x y z xw yw zw w]
            # print the simplified path
            if plot:
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(data[:, 1], data[:, 0])
                ax.invert_xaxis()
                ax.set(xlabel='y (m)', ylabel='x (m)',
                       title=u'2D Path in map - FP: {}, Time: {}s'.format(
                           verify_ompl_movement_solution(self.setup, debug=True),
                           self.setup.getLastPlanComputationTime()))
                # ax = fig.gca(projection='2d')
                # ax.plot(data[:, 0], data[:, 1], '.-')
                plt.show()
        self.setup.clear()
        return data


class NavigationPlanner(OMPLPlanner):

    def __init__(self, kitchen_floor_space, collision_scene, robot, root_link, tip_link, pose_goal, map_frame, god_map,
                 config='fast_without_refine', motion_validator_class=None):
        super(NavigationPlanner, self).__init__(False, kitchen_floor_space, collision_scene, robot, root_link, tip_link,
                                                pose_goal, map_frame, config, god_map, motion_validator_class=motion_validator_class)

    def get_planner(self, si):
        # Navigation:
        # RRTstar, RRTsharp: very sparse(= path length eq max range), high number of fp, slow relative to RRTConnect
        # RRTConnect: many points, low number of fp
        # PRMstar: no set_range function, finds no solutions
        # ABITstar: slow, sparse, but good
        # if self.fast_navigation:
        #    planner = og.RRTConnect(si)
        #    planner.setRange(0.1)
        # else:
        planner = og.RRTConnect(si)
        planner.setRange(1.0)
        # planner.setIntermediateStates(True)
        # planner.setup()
        return planner

    def plan(self, js, plot=True):

        si = self.setup.getSpaceInformation()
        self.collision_checker = GiskardRobotBulletCollisionChecker(self.is_3D, self.robot.root_link, u'base_footprint',
                                                                    self.collision_scene, dist=0.1)
        si.setStateValidityChecker(TwoDimStateValidator(si, self.collision_checker))
        self.motion_validator = self.motion_validator_class(self.collision_scene, u'base_footprint', self.robot, self.collision_checker, self.god_map, js=js)
        si.setMotionValidator(OMPLMotionValidator(si, self.is_3D, self.motion_validator))

        si.setup()
        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(allocGiskardValidStateSample))

        # Set Start and Goal pose for Planner
        start = self.get_start_state(self.space)
        goal = self.get_goal_state(self.space)
        self.setup.setStartAndGoalStates(start, goal)

        # Set Optimization Function
        # optimization_objective = PathLengthAndGoalOptimizationObjective(self.setup.getSpaceInformation(),
        #                                                                goal)
        # self.setup.setOptimizationObjective(optimization_objective)

        if not self.setup.getSpaceInformation().isValid(goal()):
            raise Exception('The given goal pose is not valid.')  # todo: make GlobalPlanningException

        self.setup.setup()
        planner_status = self.solve()
        # og.PathSimplifier(si).smoothBSpline(ss.getSolutionPath()) # takes around 20-30 secs with RTTConnect(0.1)
        # og.PathSimplifier(si).reduceVertices(ss.getSolutionPath())
        data = numpy.array([])
        if planner_status in [ob.PlannerStatus.EXACT_SOLUTION]:
            # try to shorten the path
            # ss.simplifySolution() DONT! NO! DONT UNCOMMENT THAT! NO! STOP IT! FIRST IMPLEMENT CHECKMOTION! THEN TRY AGAIN!
            # Make sure enough subpaths are available for Path Following
            self.setup.getSolutionPath().interpolate(20)
            data = ompl_states_matrix_to_np(
                self.setup.getSolutionPath().printAsMatrix())  # [[x, y, theta]]
            # print the simplified path
            if plot:
                plt.close()
                fig, ax = plt.subplots()
                ax.plot(data[:, 1], data[:, 0])
                ax.invert_xaxis()
                ax.set(xlabel='y (m)', ylabel='x (m)',
                       title=u'Navigation Path in map - FP: {}, Time: {}s'.format(
                           verify_ompl_navigation_solution(self.setup, debug=True),
                           self.setup.getLastPlanComputationTime()
                       ))
                # ax = fig.gca(projection='2d')
                # ax.plot(data[:, 0], data[:, 1], '.-')
                plt.show()
        self.setup.clear()
        return data


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
