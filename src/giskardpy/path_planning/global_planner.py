#!/usr/bin/env python

import json
import sys
import threading
from collections import namedtuple
from datetime import datetime

import numpy as np
import rospy
import tf.transformations
import yaml
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

from giskard_msgs.msg import Constraint
from py_trees import Status
import pybullet as p

from copy import deepcopy

import giskardpy.identifier as identifier
from giskardpy.exceptions import GlobalPlanningException, InfeasibleGlobalPlanningException, \
    FeasibleGlobalPlanningException, ReplanningException
from giskardpy.path_planning.motion_validator import ObjectRayMotionValidator, CompoundBoxMotionValidator
from giskardpy.path_planning.ompl_wrapper import ompl_se3_state_to_pose, \
    ompl_states_matrix_to_np, OMPLMotionValidator, is_3D, OMPLStateValidator
from giskardpy.path_planning.state_validator import GiskardRobotBulletCollisionChecker
from giskardpy.tree.behaviors.get_goal import GetGoal
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.tfwrapper import transform_pose, np_to_pose_stamped, \
    np_to_pose, pose_diff, interpolate_pose, pose_to_list

from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og
from ompl import tools as ot

# todo: put below in ros params
SolveParameters = namedtuple('SolveParameters', 'initial_solve_time refine_solve_time max_initial_iterations '
                                                'max_refine_iterations min_refine_thresh')

from giskardpy.utils.utils import convert_dictionary_to_ros_message, convert_ros_message_to_dictionary, write_to_tmp, \
    raise_to_blackboard


def allocPathLengthDirectInfSampler(probDefn, maxNumberCalls):
    return lambda: ob.PathLengthDirectInfSampler(probDefn, maxNumberCalls)


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
        rospy.loginfo(u'Num Invalid States: {}, Num Valid States: {}, Rate FP: {}'.format(f, t, f / t))
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
        rospy.loginfo(u'Num Invalid States: {}, Num Valid States: {}, Rate FP: {}'.format(f, t, f / t))
    return f / t


def allocGiskardValidStateSample(si):
    return ob.BridgeTestValidStateSampler(si)


class GlobalPlanner(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        self.supported_cart_goals = ['CartesianPose', 'CartesianPosition', 'CartesianPathCarrot', 'CartesianPreGrasp']

        # self.robot = self.robot
        self.map_frame = 'map'
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

        self.narrow = self.__goal_dict[u'narrow'] if 'narrow' in self.__goal_dict else False
        if self.narrow:
            try:
                self.narrow_padding = self.__goal_dict[u'narrow_padding']
                if self.narrow_padding < 0:
                    raise Exception('The padding value must be positive.')
            except KeyError:
                raise Exception('Please specify a narrow padding value.')

        self.navigation = self.__goal_dict[u'navigation'] if 'navigation' in self.__goal_dict else False
        self.manipulation = self.__goal_dict[u'manipulation'] if 'manipulation' in self.__goal_dict else False

        ros_pose = convert_dictionary_to_ros_message(self.__goal_dict[u'goal'])
        self.goal = transform_pose('map', ros_pose)  # type: PoseStamped

        self.root_link = self.__goal_dict[u'root_link']
        tip_link = self.__goal_dict[u'tip_link']
        links = self.world.groups[self.robot.name].link_names
        if tip_link not in links:
            raise Exception('wa')
        self.tip_link = tip_link
        link_names = self.robot.link_names

        # if self.root_link not in link_names:
        #     raise Exception(u'Root_link {} is no known link of the robot.'.format(self.root_link))
        if self.tip_link not in link_names:
            raise Exception(u'Tip_link {} is no known link of the robot.'.format(self.tip_link))
        # if not self.robot.are_linked(self.root_link, self.tip_link):
        #    raise Exception(u'Did not found link chain of the robot from'
        #                    u' root_link {} to tip_link {}.'.format(self.root_link, self.tip_link))

    def _get_navigation_planner(self, planner_name, motion_validator_type, range, time):
        self.collision_scene.update_collision_environment()
        map_frame = 'map'
        motion_validator_class = self._get_motion_validator_class(motion_validator_type)
        verify_solution_f = verify_ompl_navigation_solution
        dist = 0.1
        planner = MovementPlanner(False, planner_name, motion_validator_class, range, time,
                                  self.kitchen_floor_space, self.collision_scene,
                                  self.robot, self.root_link, self.tip_link, self.goal.pose, map_frame, self.god_map,
                                  config=self.navigation_config, dist=dist, verify_solution_f=verify_solution_f)
        return planner

    def _get_movement_planner(self, planner_name, motion_validator_type, range, time):
        sampling_goal_axis = self.__goal_dict['goal_sampling_axis'] \
            if 'goal_sampling_axis' in self.__goal_dict \
            else None
        self.collision_scene.update_collision_environment()
        map_frame = 'map'
        motion_validator_class = self._get_motion_validator_class(motion_validator_type)
        verify_solution_f = verify_ompl_movement_solution
        dist = 0
        planner = MovementPlanner(True, planner_name, motion_validator_class, range, time,
                                  self.kitchen_space, self.collision_scene,
                                  self.robot, self.root_link, self.tip_link, self.goal.pose, map_frame, self.god_map,
                                  config=self.movement_config, dist=dist, verify_solution_f=verify_solution_f)
        return planner

    def _get_narrow_movement_planner(self, planner_name, motion_validator_type, range, time):
        sampling_goal_axis = self.__goal_dict['goal_sampling_axis'] \
            if 'goal_sampling_axis' in self.__goal_dict \
            else None
        self.collision_scene.update_collision_environment()
        map_frame = 'map'
        motion_validator_class = self._get_motion_validator_class(motion_validator_type)
        verify_solution_f = verify_ompl_movement_solution
        dist = 0
        planner = NarrowMovementPlanner(planner_name, motion_validator_class, range, time,
                                        self.kitchen_space, self.collision_scene,
                                        self.robot, self.root_link, self.tip_link, self.goal.pose, map_frame,
                                        self.god_map, config=self.movement_config, dist=dist,
                                        narrow_padding=self.narrow_padding,
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
              navigation=False, movement=False, narrow=False):
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
                            planner.interpolate_solution()
                            trajectory = planner.get_solution(ob.PlannerStatus.EXACT_SOLUTION)
                        rospy.logerr(trajectory.tolist())
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

    def __check_any(self, bools):
        return sum([1 for v in bools if v]) != 1

    def plan(self, navigation=False, movement=False, narrow=False):
        if self.__check_any([navigation, movement]):
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
            predict_f = 5.0
        else:
            if narrow:
                predict_f = 2.0
            else:
                predict_f = 5.0
        return trajectory, predict_f

    def parse_cart_goal(self, cart_c):

        __goal_dict = yaml.load(cart_c.parameter_value_pair)
        ros_pose = convert_dictionary_to_ros_message(__goal_dict[u'goal'])
        pose_goals = list()
        if 'goals' in __goal_dict:
            pose_goals = list(map(convert_dictionary_to_ros_message, __goal_dict[u'goals']))
        pose_goal = transform_pose(self.map_frame, ros_pose).pose

        root_link = __goal_dict[u'root_link']
        tip_link = __goal_dict[u'tip_link']
        link_names = self.get_robot().link_names

        # if root_link not in link_names:
        #    raise Exception(u'Root_link {} is no known link of the robot.'.format(root_link))
        if tip_link not in link_names:
            raise Exception(u'Tip_link {} is no known link of the robot.'.format(tip_link))
        # if not self.get_robot().are_linked(root_link, tip_link):
        #    raise Exception(u'Did not found link chain of the robot from'
        #                    u' root_link {} to tip_link {}.'.format(root_link, tip_link))

        return root_link, tip_link, pose_goal, pose_goals

    def is_unplanned(self, cartesian_constraint):
        if cartesian_constraint.type == 'CartesianPathCarrot':
            r, t, p, gs = self.parse_cart_goal(cartesian_constraint)
            return len(gs) == 0
        else:
            return False

    @profile
    def update(self):

        # global_planner_needed = self.god_map.get_data(identifier.global_planner_needed)
        # if not global_planner_needed:
        #    return Status.SUCCESS

        # Check if move_cmd exists
        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.SUCCESS

        # Check if move_cmd contains a Cartesian Goal
        cart_c = self.get_cart_goal(move_cmd)
        if not cart_c:
            return Status.SUCCESS

        if not self.is_unplanned(cart_c):
            return Status.SUCCESS

        # Parse and save the Cartesian Goal Constraint
        self.save_cart_goal(cart_c)
        try:
            trajectory, predict_f = self.plan(navigation=self.navigation,
                                              movement=self.manipulation,
                                              narrow=self.narrow)
        except FeasibleGlobalPlanningException:
            raise_to_blackboard(GlobalPlanningException())
            return Status.FAILURE
        poses = []
        start = None
        for i, point in enumerate(trajectory):
            base_pose = PoseStamped()
            base_pose.header.frame_id = 'map'
            base_pose.pose.position.x = point[0]
            base_pose.pose.position.y = point[1]
            base_pose.pose.position.z = point[2] if len(point) > 3 else 0
            if len(point) > 3:
                base_pose.pose.orientation = Quaternion(point[3], point[4], point[5], point[6])
            else:
                arr = tf.transformations.quaternion_from_euler(0, 0, point[2])
                base_pose.pose.orientation = Quaternion(arr[0], arr[1], arr[2], arr[3])
            if i == 0:
                # important assumption for constraint:
                # we do not to reach the first pose, since it is the start pose
                start = base_pose
            else:
                poses.append(base_pose)
        # poses[-1].pose.orientation = self.goal.pose.orientation
        move_cmd.constraints.remove(cart_c)
        move_cmd.constraints.append(self.get_cartesian_path_constraints(start, poses, predict_f))
        self.get_god_map().set_data(identifier.next_move_goal, move_cmd)
        # self.get_god_map().set_data(identifier.global_planner_needed, False)
        return Status.SUCCESS

    def get_cartesian_path_constraints(self, start, poses, predict_f):

        d = dict()
        d[u'parameter_value_pair'] = deepcopy(self.__goal_dict)

        goal = self.goal
        goal.header.stamp = rospy.Time(0)

        c_d = deepcopy(d)
        c_d[u'parameter_value_pair'][u'start'] = convert_ros_message_to_dictionary(start)
        c_d[u'parameter_value_pair'][u'goal'] = convert_ros_message_to_dictionary(goal)
        c_d[u'parameter_value_pair'][u'goals'] = list(map(convert_ros_message_to_dictionary, poses))
        if 'predict_f' not in c_d[u'parameter_value_pair']:
            c_d[u'parameter_value_pair'][u'predict_f'] = predict_f

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
        bounds.setLow(0, -4)
        bounds.setHigh(0, 2)
        bounds.setLow(1, -3)
        bounds.setHigh(1, 4)
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
        # refills lab
        # bounds.setLow(0, -3)
        # bounds.setHigh(0, 5)
        # bounds.setLow(1, -4)
        # bounds.setHigh(1, 5)
        # kitchen
        bounds.setLow(0, -4)
        bounds.setHigh(0, 2)
        bounds.setLow(1, -3)
        bounds.setHigh(1, 4)
        space.setBounds(bounds)

        return space


class OMPLPlanner(object):

    def __init__(self, is_3D, planner_name, motion_validator_class, range, time, space,
                 collision_scene, robot, root_link, tip_link, pose_goal, map_frame, config, god_map,
                 verify_solution_f=None, dist=0.0):
        self.setup = None
        self.plot = god_map.get_data(identifier.plot_path)
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
        n = f"test_ease_fridge_pregrasp_1-" \
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
        # req.maxMem = 100.0
        req.runCount = 5
        req.displayProgress = True
        req.simplify = False
        b.benchmark(req)
        b.saveResultsToFile()

    def solve(self):
        # Get solve parameters
        planner_name = self.setup.getPlanner().getName()
        # discrete_checking = self.motion_validator_class is None
        if planner_name not in self._planner_solve_params:
            solve_params = self._planner_solve_params[None][self.config]
        else:
            solve_params = self._planner_solve_params[planner_name][self.config]
        # debugging = sys.gettrace() is not None
        # initial_solve_time = solve_params.initial_solve_time if not debugging else solve_params.initial_solve_time * debug_factor
        # initial_solve_time = initial_solve_time * discrete_factor if discrete_checking else initial_solve_time
        initial_solve_time = self.max_time  # min(initial_solve_time, self.max_time)
        # refine_solve_time = solve_params.refine_solve_time if not debugging else solve_params.refine_solve_time * debug_factor
        # refine_solve_time = refine_solve_time * discrete_factor if discrete_checking else refine_solve_time
        max_initial_iterations = solve_params.max_initial_iterations
        # max_refine_iterations = solve_params.max_refine_iterations
        # min_refine_thresh = solve_params.min_refine_thresh
        max_initial_solve_time = min(initial_solve_time * max_initial_iterations, self.max_time)
        # max_refine_solve_time = refine_solve_time * max_refine_iterations

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
            # self.shorten_path(path, self.goal)
        except RuntimeError:
            raise Exception('Problem Definition in Setup may have changed..')
        return path


class MovementPlanner(OMPLPlanner):

    def __init__(self, is_3D, planner_name, motion_validator_class, range, time, kitchen_space,
                 collision_scene, robot, root_link, tip_link, pose_goal, map_frame, god_map,
                 config='slow_without_refine', verify_solution_f=None, dist=0.0):
        super(MovementPlanner, self).__init__(is_3D, planner_name, motion_validator_class, range,
                                              time, kitchen_space, collision_scene, robot, root_link, tip_link,
                                              pose_goal, map_frame, config, god_map,
                                              verify_solution_f=verify_solution_f, dist=dist)

    def clear(self):
        super().clear()
        if self.motion_validator_class is not None:
            self.motion_validator.clear()
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
                                                                    self.collision_scene, self.god_map)
        si.setStateValidityChecker(OMPLStateValidator(si, self.is_3D, self.collision_checker))

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
        # si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(allocGiskardValidStateSample))
        si.setup()

        self.start = self.get_start_state(self.space)
        self.goal = self.get_goal_state(self.space)

        if not si.isValid(self.start()):
            rospy.logerr('Start is not valid')
            raise InfeasibleGlobalPlanningException()
        if not si.isValid(self.goal()):
            rospy.logerr('Start is not valid')
            raise InfeasibleGlobalPlanningException()

        self.setup.setStartAndGoalStates(self.start, self.goal)

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
                path.interpolate(int(path_cost / 0.025))
            else:
                path.interpolate(int(path_cost / 0.05))
        else:
            path.interpolate(int(path_cost / 0.2))
        if self.verify_solution_f is not None:
            if self.verify_solution_f(self.setup, self.get_solution_path(), debug=True) != 0:
                rospy.loginfo('Interpolated path is invalid. Going to re-plan...')
                raise ReplanningException('Interpolated Path is invalid.')
        else:
            rospy.logwarn('Interpolated path returned is not validated.')

    def get_solution(self, planner_status):
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
            if self.plot:
                self.plot_solution(data)
        return data

    def plot_solution(self, data, debug=True):
        plt.close()
        dim = '3D' if self.is_3D else '2D'
        if debug:
            self.verify_solution_f(self.setup, self.get_solution_path(), debug=debug)
        if self.verify_solution_f is not None:
            verify = ' Time: {}s'.format(self.setup.getLastPlanComputationTime())
        else:
            verify = ''
        path = self.setup.getSolutionPath()
        path_cost = path.cost(self.optimization_objective).value()
        self.god_map.set_data(identifier.giskard + ['path_time'], self.setup.getLastPlanComputationTime())
        self.god_map.set_data(identifier.giskard + ['path_cost'], path_cost)
        cost = '- Cost: {}'.format(str(round(path_cost, 5)))
        title = u'{} Path from {} in map\n{} {}'.format(dim, self.setup.getPlanner().getName(), verify, cost)
        fig, ax = plt.subplots()
        ax.plot(data[:, 1], data[:, 0])
        ax.invert_xaxis()
        ax.set(xlabel='y (m)', ylabel='x (m)',
               title=title)
        # ax = fig.gca(projection='2d')
        # ax.plot(data[:, 0], data[:, 1], '.-')
        # plt.savefig('/home/thomas/master_thesis/benchmarking_data/'
        #            'path_following/box/navi_5/{}_path_planner.png'.format(rospy.get_time()))
        plt.show()


class NarrowMovementPlanner(MovementPlanner):
    def __init__(self, planner_name, motion_validator_class, range, time, kitchen_space,
                 collision_scene, robot, root_link, tip_link, pose_goal, map_frame, god_map,
                 config='slow_without_refine', dist=0.1, narrow_padding=1.0, sampling_goal_axis=None,
                 verify_solution_f=None):
        super(NarrowMovementPlanner, self).__init__(True, planner_name,
                                                    motion_validator_class, range, time, kitchen_space,
                                                    collision_scene, robot, root_link, tip_link,
                                                    pose_goal, map_frame, god_map, config=config, dist=dist,
                                                    verify_solution_f=verify_solution_f)
        self.sampling_goal_axis = sampling_goal_axis
        self.narrow_padding = narrow_padding
        self.reversed_start_and_goal = False
        self.directional_planner = [
            'RRT', 'TRRT', 'LazyRRT',
            # 'EST','KPIECE1', 'BKPIECE1', 'LBKPIECE1', 'FMT',
            # 'STRIDE',
            # 'BITstar', 'ABITstar', 'kBITstar', 'kABITstar',
            'RRTstar', 'LBTRRT',
            # 'SST',
            'RRTXstatic', 'RRTsharp', 'RRT#', 'InformedRRTstar',
            # 'SORRTstar'
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

    def create_goal_specific_space(self):

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
            bounds.setLow(0, min(s_x, g_x) - self.narrow_padding)
            bounds.setHigh(0, max(s_x, g_x) + self.narrow_padding)
            bounds.setLow(1, min(s_y, g_y) - self.narrow_padding)
            bounds.setHigh(1, max(s_y, g_y) + self.narrow_padding)
            if is_3D(self.space):
                bounds.setLow(2, min(s_z, g_z) - self.narrow_padding)
                bounds.setHigh(2, max(s_z, g_z) + self.narrow_padding)

            # Save it
            self.space.setBounds(bounds)

    def get_solution(self, planner_status):
        data = super().get_solution(planner_status)
        if len(data) > 0 and self.planner_name in self.directional_planner:

            if not self.reversed_start_and_goal:
                end_i = self.pose_goal
            else:
                end_i = ompl_se3_state_to_pose(self.start())
            begin_i = Pose()
            begin_i.position.x = data[-1][0]
            begin_i.position.y = data[-1][1]
            begin_i.position.z = data[-1][2]
            begin_i.orientation.x = data[-1][3]
            begin_i.orientation.y = data[-1][4]
            begin_i.orientation.z = data[-1][5]
            begin_i.orientation.w = data[-1][6]
            diff = pose_diff(begin_i, end_i)
            segments = diff / self.range
            rospy.logerr(segments)
            for i in range(1, int(segments)):
                p = pose_to_list(interpolate_pose(begin_i, end_i, i * (1 / segments)))
                data = np.append(data, [np.append(p[0], p[1])], axis=0)

            if not self.reversed_start_and_goal:
                data = np.append(data, [np.append(pose_to_list(self.pose_goal)[0], pose_to_list(self.pose_goal)[1])],
                                 axis=0)
            else:
                data = np.append(data, [np.append(pose_to_list(ompl_se3_state_to_pose(self.start()))[0],
                                                  pose_to_list(ompl_se3_state_to_pose(self.start()))[1])], axis=0)
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