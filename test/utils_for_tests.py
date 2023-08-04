import csv
import keyword
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Queue
from time import time
from typing import Tuple, Optional, List, Type

import actionlib
import control_msgs
import hypothesis.strategies as st
import numpy as np
import rospy
from angles import shortest_angular_distance
from control_msgs.msg import FollowJointTrajectoryActionGoal
from geometry_msgs.msg import PoseStamped, Point, Vector3Stamped, PointStamped, QuaternionStamped
from hypothesis import assume
from hypothesis.strategies import composite
from numpy import pi
from rospy import Timer
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_py import LookupException, ExtrapolationException
from visualization_msgs.msg import Marker

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import CollisionEntry, MoveResult, MoveGoal
from giskard_msgs.srv import UpdateWorldResponse, DyeGroupResponse
from giskardpy import identifier

from giskardpy.configs.behavior_tree_config import BehaviorTreeConfig
from giskardpy.configs.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.qp_controller_config import QPControllerConfig, SupportedQPSolver
from giskardpy.configs.robot_interface_config import RobotInterfaceConfig
from giskardpy.configs.world_config import WorldConfig
from giskardpy.data_types import KeyDefaultDict, JointStates
from giskardpy.god_map_user import GodMapWorshipper
from giskardpy.model.collision_world_syncer import Collisions, Collision
from giskardpy.my_types import PrefixName, Derivatives
from giskardpy.exceptions import UnknownGroupException
from giskardpy.goals.goal import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.god_map import GodMap
from giskardpy.model.joints import OneDofJoint, OmniDrive, DiffDrive
from giskardpy.model.world import WorldTree
from giskardpy.python_interface import GiskardWrapper
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.qp_controller import available_solvers
from giskardpy.tree.behaviors.plot_debug_expressions import PlotDebugExpressions
from giskardpy.tree.behaviors.plot_trajectory import PlotTrajectory
from giskardpy.tree.behaviors.visualization import VisualizationBehavior
from giskardpy.tree.garden import TreeManager, ControlModes
from giskardpy.utils import logging, utils
from giskardpy.utils.math import compare_poses
from giskardpy.utils.utils import msg_to_list, resolve_ros_iris
import os

BIG_NUMBER = 1e100
SMALL_NUMBER = 1e-100

folder_name = 'tmp_data/'


def vector(x):
    return st.lists(float_no_nan_no_inf(), min_size=x, max_size=x)


update_world_error_codes = {value: name for name, value in vars(UpdateWorldResponse).items() if
                            isinstance(value, int) and name[0].isupper()}


def update_world_error_code(code):
    return update_world_error_codes[code]


move_result_error_codes = {value: name for name, value in vars(MoveResult).items() if
                           isinstance(value, int) and name[0].isupper()}


def move_result_error_code(code):
    return move_result_error_codes[code]


def robot_urdfs():
    return st.sampled_from(['urdfs/pr2.urdfs', 'urdfs/boxy.urdfs', 'urdfs/iai_donbot.urdfs'])


def angle_positive():
    return st.floats(0, 2 * np.pi)


def random_angle():
    return st.floats(-np.pi, np.pi)


def keys_values(max_length=10, value_type=st.floats(allow_nan=False)):
    return lists_of_same_length([variable_name(), value_type], max_length=max_length, unique=True)


def compare_axis_angle(actual_angle, actual_axis, expected_angle, expected_axis, decimal=3):
    try:
        np.testing.assert_array_almost_equal(actual_axis, expected_axis, decimal=decimal)
        np.testing.assert_almost_equal(shortest_angular_distance(actual_angle, expected_angle), 0, decimal=decimal)
    except AssertionError:
        try:
            np.testing.assert_array_almost_equal(actual_axis, -expected_axis, decimal=decimal)
            np.testing.assert_almost_equal(shortest_angular_distance(actual_angle, abs(expected_angle - 2 * pi)), 0,
                                           decimal=decimal)
        except AssertionError:
            np.testing.assert_almost_equal(shortest_angular_distance(actual_angle, 0), 0, decimal=decimal)
            np.testing.assert_almost_equal(shortest_angular_distance(0, expected_angle), 0, decimal=decimal)
            assert not np.any(np.isnan(actual_axis))
            assert not np.any(np.isnan(expected_axis))


@composite
def variable_name(draw):
    variable = draw(st.text('qwertyuiopasdfghjklzxcvbnm', min_size=1))
    assume(variable not in keyword.kwlist)
    return variable


@composite
def lists_of_same_length(draw, data_types=(), min_length=1, max_length=10, unique=False):
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    lists = []
    for elements in data_types:
        lists.append(draw(st.lists(elements, min_size=length, max_size=length, unique=unique)))
    return lists


@composite
def rnd_joint_state(draw, joint_limits):
    return {jn: draw(st.floats(ll, ul, allow_nan=False, allow_infinity=False)) for jn, (ll, ul) in joint_limits.items()}


@composite
def rnd_joint_state2(draw, joint_limits):
    muh = draw(joint_limits)
    muh = {jn: ((ll if ll is not None else pi * 2), (ul if ul is not None else pi * 2))
           for (jn, (ll, ul)) in muh.items()}
    return {jn: draw(st.floats(ll, ul, allow_nan=False, allow_infinity=False)) for jn, (ll, ul) in muh.items()}


@composite
def pr2_joint_state(draw):
    pass


def pr2_urdf():
    path = resolve_ros_iris('package://giskardpy/test/urdfs/pr2_with_base.urdf')
    with open(path, 'r') as f:
        urdf_string = f.read()
    return urdf_string


def pr2_without_base_urdf():
    with open('urdfs/pr2.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def base_bot_urdf():
    with open('urdfs/2d_base_bot.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def donbot_urdf():
    with open('urdfs/iai_donbot.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def boxy_urdf():
    with open('urdfs/boxy.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def hsr_urdf():
    with open('urdfs/hsr.urdf', 'r') as f:
        urdf_string = f.read()
    return urdf_string


def float_no_nan_no_inf(outer_limit=1e5):
    return float_no_nan_no_inf_min_max(-outer_limit, outer_limit)


def float_no_nan_no_inf_min_max(min_value=-1e5, max_value=1e5):
    return st.floats(allow_nan=False, allow_infinity=False, max_value=max_value, min_value=min_value,
                     allow_subnormal=False)


@composite
def sq_matrix(draw):
    i = draw(st.integers(min_value=1, max_value=10))
    i_sq = i ** 2
    l = draw(st.lists(float_no_nan_no_inf(outer_limit=1000), min_size=i_sq, max_size=i_sq))
    return np.array(l).reshape((i, i))


def unit_vector(length, elements=None):
    if elements is None:
        elements = float_no_nan_no_inf()
    vector = st.lists(elements,
                      min_size=length,
                      max_size=length).filter(lambda x: SMALL_NUMBER < np.linalg.norm(x) < BIG_NUMBER)

    def normalize(v):
        v = [round(x, 4) for x in v]
        l = np.linalg.norm(v)
        if l == 0:
            return np.array([0] * (length - 1) + [1])
        return np.array([x / l for x in v])

    return st.builds(normalize, vector)


def quaternion(elements=None):
    return unit_vector(4, elements)


def pykdl_frame_to_numpy(pykdl_frame):
    return np.array([[pykdl_frame.M[0, 0], pykdl_frame.M[0, 1], pykdl_frame.M[0, 2], pykdl_frame.p[0]],
                     [pykdl_frame.M[1, 0], pykdl_frame.M[1, 1], pykdl_frame.M[1, 2], pykdl_frame.p[1]],
                     [pykdl_frame.M[2, 0], pykdl_frame.M[2, 1], pykdl_frame.M[2, 2], pykdl_frame.p[2]],
                     [0, 0, 0, 1]])


class GiskardTestWrapper(GiskardWrapper, GodMapWorshipper):
    god_map: GodMap
    default_pose = {}
    better_pose = {}
    odom_root = 'odom'
    tree: TreeManager

    def __init__(self,
                 giskard: Giskard):
        self.total_time_spend_giskarding = 0
        self.total_time_spend_moving = 0
        self._alive = True

        try:
            from iai_naive_kinematics_sim.srv import UpdateTransform
            self.set_localization_srv = rospy.ServiceProxy('/map_odom_transform_publisher/update_map_odom_transform',
                                                           UpdateTransform)
        except Exception as e:
            self.set_localization_srv = None

        self.giskard = giskard
        self.giskard.grow()
        if 'GITHUB_WORKFLOW' in os.environ:
            logging.loginfo('Inside github workflow, turning off visualization')
            plugins_to_disable = [VisualizationBehavior, PlotTrajectory, PlotDebugExpressions]
            for behavior_type in plugins_to_disable:
                for node in self.tree_manager.get_nodes_of_type(behavior_type):
                    self.tree_manager.disable_node(node.name)
        if 'QP_SOLVER' in os.environ:
            self.giskard.qp_controller_config.set_qp_solver(SupportedQPSolver[os.environ['QP_SOLVER']])
        # self.tree_manager = TreeManager.from_param_server(robot_names, namespaces)
        self.heart = Timer(period=rospy.Duration(self.tree_manager.tick_rate), callback=self.heart_beat)
        # self.namespaces = namespaces
        self.robot_names = [list(self.world.groups.keys())[0]]
        super().__init__(node_name='tests')
        self.results = Queue(100)
        self.default_root = str(self.world.root_link_name)
        self.goal_checks = defaultdict(list)

        def create_publisher(topic):
            p = rospy.Publisher(topic, JointState, queue_size=10)
            rospy.sleep(.2)
            return p

        self.joint_state_publisher = KeyDefaultDict(create_publisher)
        # rospy.sleep(1)
        self.original_number_of_links = len(self.world.links)

    def is_standalone(self):
        return self.tree_manager.control_mode == ControlModes.standalone

    def has_odometry_joint(self, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.robot_name
        parent_joint_name = self.world.groups[group_name].root_link.parent_joint_name
        if parent_joint_name is None:
            return False
        joint = self.world.get_joint(parent_joint_name)
        return isinstance(joint, (OmniDrive, DiffDrive))

    def set_seed_odometry(self, base_pose, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.robot_name
        self.set_json_goal('SetOdometry',
                           group_name=group_name,
                           base_pose=base_pose)

    def set_seed_configuration(self, seed_configuration):
        self.set_json_goal('SetSeedConfiguration',
                           seed_configuration=seed_configuration)

    def set_localization(self, map_T_odom: PoseStamped):
        if self.set_localization_srv is not None:
            from iai_naive_kinematics_sim.srv import UpdateTransformRequest
            map_T_odom.pose.position.z = 0
            req = UpdateTransformRequest()
            req.transform.translation = map_T_odom.pose.position
            req.transform.rotation = map_T_odom.pose.orientation
            assert self.set_localization_srv(req).success
            self.wait_heartbeats(15)
            p2 = self.world.compute_fk_pose(self.world.root_link_name, self.odom_root)
            compare_poses(p2.pose, map_T_odom.pose)

    def transform_msg(self, target_frame, msg, timeout=1):
        result_msg = deepcopy(msg)
        try:
            if not self.is_standalone():
                return tf.transform_msg(target_frame, result_msg, timeout=timeout)
            else:
                raise LookupException('just to trigger except block')
        except (LookupException, ExtrapolationException) as e:
            target_frame = self.world.search_for_link_name(target_frame)
            try:
                result_msg.header.frame_id = self.world.search_for_link_name(result_msg.header.frame_id)
            except UnknownGroupException:
                pass
            return self.world.transform_msg(target_frame, result_msg)

    def wait_heartbeats(self, number=2):
        behavior_tree = self.tree_manager.tree
        c = behavior_tree.count
        while behavior_tree.count < c + number:
            rospy.sleep(0.001)

    @property
    def collision_scene(self):
        """
        :rtype: giskardpy.model.collision_world_syncer.CollisionWorldSynchronizer
        """
        return self.god_map.unsafe_get_data(identifier.collision_scene)

    def get_robot(self, group_name):
        """
        :rtype: giskardpy.model.world.WorldBranch
        """
        return self.world.groups[group_name]

    def dye_group(self, group_name: str, rgba: Tuple[float, float, float, float],
                  expected_error_codes=(DyeGroupResponse.SUCCESS,)):
        res = super().dye_group(group_name, rgba)
        assert res.error_codes in expected_error_codes

    def heart_beat(self, timer_thing):
        if self._alive:
            self.tree_manager.tick()

    def stop_ticking(self):
        self._alive = False

    def restart_ticking(self):
        self._alive = True

    def print_qp_solver_times(self):
        with open('benchmark.csv', mode='w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['solver',
                                'filtered_variables',
                                'variables',
                                'eq_constraints',
                                'neq_constraints',
                                'num_eq_slack_variables',
                                'num_neq_slack_variables',
                                'num_slack_variables',
                                'max_derivative',
                                'data'])

            for solver_id, solver_class in available_solvers.items():
                times = solver_class.get_solver_times()
                for (filtered_variables, variables, eq_constraints, neq_constraints, num_eq_slack_variables,
                     num_neq_slack_variables, num_slack_variables), times in sorted(times.items()):
                    csvwriter.writerow([solver_id.name,
                                        str(filtered_variables),
                                        str(variables),
                                        str(eq_constraints),
                                        str(neq_constraints),
                                        str(num_eq_slack_variables),
                                        str(num_neq_slack_variables),
                                        str(num_slack_variables),
                                        str(int(self.god_map.get_data(identifier.max_derivative))),
                                        str(times)])

        logging.loginfo('saved benchmark file')

    def tear_down(self):
        self.print_qp_solver_times()
        rospy.sleep(1)
        self.heart.shutdown()
        # TODO it is strange that I need to kill the services... should be investigated. (:
        self.tree_manager.kill_all_services()
        giskarding_time = self.total_time_spend_giskarding
        if self.god_map.get_data(identifier.control_mode) != ControlModes.standalone:
            giskarding_time -= self.total_time_spend_moving
        logging.loginfo(f'total time spend giskarding: {giskarding_time}')
        logging.loginfo(f'total time spend moving: {self.total_time_spend_moving}')
        logging.loginfo('stopping tree')

    def set_object_joint_state(self, object_name, joint_state):
        super().set_object_joint_state(object_name, joint_state)
        self.wait_heartbeats(3)
        current_js = self.world.groups[object_name].state
        joint_names_with_prefix = set(j.long_name for j in current_js)
        joint_state_names = list()
        for j_n in joint_state.keys():
            if type(j_n) == PrefixName or '/' in j_n:
                joint_state_names.append(j_n)
            else:
                joint_state_names.append(str(PrefixName(j_n, object_name)))
        assert set(joint_state_names).difference(joint_names_with_prefix) == set()
        for joint_name, state in current_js.items():
            if joint_name.short_name in joint_state:
                np.testing.assert_almost_equal(state.position, joint_state[joint_name.short_name], 2)

    def set_kitchen_js(self, joint_state):
        if self.is_standalone():
            self.set_seed_configuration(joint_state)
            self.allow_all_collisions()
            self.plan_and_execute()
        else:
            self.set_object_joint_state(self.kitchen_name, joint_state)

    def set_apartment_js(self, joint_state):
        if self.is_standalone():
            self.set_seed_configuration(joint_state)
            self.allow_all_collisions()
            self.plan_and_execute()
        else:
            self.set_object_joint_state(self.environment_name, joint_state)

    def compare_joint_state(self, current_js: dict, goal_js: dict, decimal: int = 2):
        for joint_name in goal_js:
            goal = goal_js[joint_name]
            current = current_js[joint_name]
            joint_name = self.world.search_for_joint_name(joint_name)
            if self.world.is_joint_continuous(joint_name):
                np.testing.assert_almost_equal(shortest_angular_distance(goal, current), 0, decimal=decimal,
                                               err_msg='{}: actual: {} desired: {}'.format(joint_name, current,
                                                                                           goal))
            else:
                np.testing.assert_almost_equal(current, goal, decimal,
                                               err_msg='{}: actual: {} desired: {}'.format(joint_name, current,
                                                                                           goal))

    def get_robot_short_root_link_name(self, root_group: str = None):
        # If robots exist
        if len(self.world.robot_names) != 0:
            # If a group is given, just return the root_link_name of the SubTreeWorld
            if root_group is not None:
                root_link = self.world.groups[root_group].root_link_name
            else:
                # If only one robot is imported
                if len(self.world.robot_names) == 1:
                    root_link = self.world.groups[self.world.robot_names[0]].root_link_name
                else:
                    raise Exception('Multiple Robots detected: root group is needed'
                                    ' to get the root link automatically.')
            return root_link.short_name if type(root_link) == PrefixName else root_link

    def get_root_and_tip_link(self, root_link: str, tip_link: str,
                              root_group: str = None, tip_group: str = None) -> Tuple[PrefixName, PrefixName]:
        return self.world.search_for_link_name(root_link, root_group), \
            self.world.search_for_link_name(tip_link, tip_group)

    #
    # GOAL STUFF #################################################################################################
    #

    def set_joint_goal(self, goal, weight=None, hard=False, decimal=2, expected_error_codes=(MoveResult.SUCCESS,),
                       check=True, group_name=None):
        super().set_joint_goal(goal, group_name, weight=weight, hard=hard)
        if check:
            self.add_goal_check(JointGoalChecker(giskard=self,
                                                 goal_state=goal,
                                                 decimal=decimal))

    def teleport_base(self, goal_pose):
        pass
        # goal_pose = tf.transform_pose(self.default_root, goal_pose)
        # js = {'odom_x_joint': goal_pose.pose.position.x,
        #       'odom_y_joint': goal_pose.pose.position.y,
        #       'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
        #                                                               goal_pose.pose.orientation.y,
        #                                                               goal_pose.pose.orientation.z,
        #                                                               goal_pose.pose.orientation.w]))[0]}
        # goal = SetJointStateRequest()
        # goal.state = position_dict_to_joint_states(js)
        # # self.set_base.call(goal)
        # rospy.sleep(0.5)

    def set_rotation_goal(self,
                          goal_orientation: QuaternionStamped,
                          tip_link: str,
                          root_link: Optional[str] = None,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          weight: Optional[float] = None,
                          max_velocity: Optional[float] = None,
                          check: bool = False,
                          **kwargs):
        if root_link is None:
            root_link = self.world.root_link_name
            root_group = None
        super().set_rotation_goal(goal_orientation=goal_orientation,
                                  tip_link=tip_link,
                                  tip_group=tip_group,
                                  root_link=root_link,
                                  root_group=root_group,
                                  max_velocity=max_velocity,
                                  weight=weight, **kwargs)
        if check:
            full_root_link, full_tip_link = self.get_root_and_tip_link(root_link=root_link, root_group=root_group,
                                                                       tip_link=tip_link, tip_group=tip_group)
            self.add_goal_check(RotationGoalChecker(self, full_tip_link, full_root_link, goal_orientation))

    def set_translation_goal(self, goal_point, tip_link, root_link=None, tip_group=None, root_group=None,
                             weight=None, max_velocity=None, check=False,
                             **kwargs):
        if root_link is None:
            root_link = self.world.root_link_name
            root_group = None
        super().set_translation_goal(goal_point=goal_point,
                                     tip_link=tip_link,
                                     tip_group=tip_group,
                                     root_link=root_link,
                                     root_group=root_group,
                                     max_velocity=max_velocity,
                                     weight=weight,
                                     **kwargs)
        if check:
            full_root_link, full_tip_link = self.get_root_and_tip_link(root_link=root_link, root_group=root_group,
                                                                       tip_link=tip_link, tip_group=tip_group)
            self.add_goal_check(TranslationGoalChecker(giskard=self,
                                                       tip_link=full_tip_link,
                                                       root_link=full_root_link,
                                                       expected=goal_point))

    def set_straight_translation_goal(self, goal_pose, tip_link, root_link=None, tip_group=None, root_group=None,
                                      weight=None, max_velocity=None,
                                      **kwargs):
        if root_link is None:
            root_link = self.world.root_link_name
            root_group = None
        super().set_straight_translation_goal(goal_pose=goal_pose,
                                              tip_link=tip_link,
                                              root_link=root_link,
                                              tip_group=tip_group,
                                              root_group=root_group,
                                              max_velocity=max_velocity, weight=weight,
                                              **kwargs)

    def set_cart_goal(self, goal_pose, tip_link, root_link=None, tip_group=None, root_group=None, weight=None,
                      linear_velocity=None,
                      angular_velocity=None, check=True, **kwargs):
        goal_point = PointStamped()
        goal_point.header = goal_pose.header
        goal_point.point = goal_pose.pose.position
        self.set_translation_goal(goal_point=goal_point,
                                  tip_link=tip_link,
                                  tip_group=tip_group,
                                  root_link=root_link,
                                  root_group=root_group,
                                  weight=weight,
                                  max_velocity=linear_velocity,
                                  check=check,
                                  **kwargs)
        goal_orientation = QuaternionStamped()
        goal_orientation.header = goal_pose.header
        goal_orientation.quaternion = goal_pose.pose.orientation
        self.set_rotation_goal(goal_orientation=goal_orientation,
                               tip_link=tip_link,
                               tip_group=tip_group,
                               root_link=root_link,
                               root_group=root_group,
                               weight=weight,
                               max_velocity=angular_velocity,
                               check=check,
                               **kwargs)

    def set_move_base_goal(self, goal_pose, check=True):
        self.set_json_goal(constraint_type='PR2DiffDriveBaseGoal',
                           goal_pose=goal_pose)
        if check:
            full_root_link = self.default_root
            full_tip_link = self.world.get_link_name('base_footprint')
            goal_point = PointStamped()
            goal_point.header = goal_pose.header
            goal_point.point = goal_pose.pose.position
            self.add_goal_check(TranslationGoalChecker(giskard=self,
                                                       tip_link=full_tip_link,
                                                       root_link=full_root_link,
                                                       expected=goal_point))
            goal_orientation = QuaternionStamped()
            goal_orientation.header = goal_pose.header
            goal_orientation.quaternion = goal_pose.pose.orientation
            self.add_goal_check(RotationGoalChecker(self, full_tip_link, full_root_link, goal_orientation))

    def set_diff_drive_base_goal(self, goal_pose, tip_link=None, root_link=None, weight=None, linear_velocity=None,
                                 angular_velocity=None, check=True, **kwargs):
        if tip_link is None:
            tip_link = 'base_footprint'
        if root_link is None:
            root_link = self.default_root
        self.set_json_goal(constraint_type='DiffDriveBaseGoal',
                           tip_link=tip_link,
                           root_link=root_link,
                           goal_pose=goal_pose,
                           max_linear_velocity=linear_velocity,
                           max_angular_velocity=angular_velocity,
                           weight=weight)
        if check:
            goal_point = PointStamped()
            goal_point.header = goal_pose.header
            goal_point.point = goal_pose.pose.position
            self.add_goal_check(TranslationGoalChecker(self, tip_link, root_link, goal_point))
            goal_orientation = QuaternionStamped()
            goal_orientation.header = goal_pose.header
            goal_orientation.quaternion = goal_pose.pose.orientation
            self.add_goal_check(RotationGoalChecker(self, tip_link, root_link, goal_orientation))

    def set_keep_hand_in_workspace(self, tip_link, map_frame=None, base_footprint=None):
        self.set_json_goal('KeepHandInWorkspace',
                           tip_link=tip_link,
                           map_frame=map_frame,
                           base_footprint=base_footprint)

    def set_diff_drive_tangential_to_point(self, goal_point: PointStamped, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        self.set_json_goal('DiffDriveTangentialToPoint',
                           goal_point=goal_point,
                           weight=weight,
                           **kwargs)

    def set_pointing_goal(self,
                          goal_point: PointStamped,
                          tip_link: str,
                          pointing_axis: Vector3Stamped,
                          root_link: str,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          max_velocity: float = 0.3,
                          weight: float = WEIGHT_BELOW_CA,
                          check=False):
        if root_link is None:
            root_link = self.world.root_link_name
            root_group = None
        super().set_pointing_goal(goal_point=goal_point,
                                  tip_link=tip_link,
                                  pointing_axis=pointing_axis,
                                  root_link=root_link,
                                  tip_group=tip_group,
                                  root_group=root_group,
                                  max_velocity=max_velocity,
                                  weight=weight)
        if check:
            full_root_link, full_tip_link = self.get_root_and_tip_link(root_link=root_link, root_group=root_group,
                                                                       tip_link=tip_link, tip_group=tip_group)
            self.add_goal_check(PointingGoalChecker(self,
                                                    tip_link=full_tip_link,
                                                    goal_point=goal_point,
                                                    root_link=full_root_link,
                                                    pointing_axis=pointing_axis))

    def set_align_planes_goal(self, tip_link, tip_normal, root_link=None, tip_group=None, root_group=None,
                              goal_normal=None, max_angular_velocity=None,
                              weight=None, check=False):
        if root_link is None:
            root_link = self.world.root_link_name
            root_group = None
        super().set_align_planes_goal(tip_link=tip_link,
                                      tip_group=tip_group,
                                      root_link=root_link,
                                      root_group=root_group,
                                      tip_normal=tip_normal,
                                      goal_normal=goal_normal,
                                      max_angular_velocity=max_angular_velocity,
                                      weight=weight)
        if check:
            full_root_link, full_tip_link = self.get_root_and_tip_link(root_link=root_link, root_group=root_group,
                                                                       tip_link=tip_link, tip_group=tip_group)
            self.add_goal_check(
                AlignPlanesGoalChecker(self, full_tip_link, tip_normal, full_root_link, goal_normal))

    def add_goal_check(self, goal_checker):
        self.goal_checks[self.number_of_cmds - 1].append(goal_checker)

    def set_straight_cart_goal(self,
                               goal_pose: PoseStamped,
                               tip_link: str,
                               root_link: str,
                               tip_group: Optional[str] = None,
                               root_group: Optional[str] = None,
                               max_linear_velocity: Optional[float] = None,
                               max_angular_velocity: Optional[float] = None,
                               reference_linear_velocity: Optional[float] = None,
                               reference_angular_velocity: Optional[float] = None,
                               weight: Optional[float] = None,
                               check=False):
        if root_link is None:
            root_link = self.world.root_link_name
            root_group = None
        super().set_straight_cart_goal(goal_pose=goal_pose,
                                       tip_link=tip_link,
                                       root_link=root_link,
                                       tip_group=tip_group,
                                       root_group=root_group,
                                       max_linear_velocity=max_linear_velocity,
                                       max_angular_velocity=max_angular_velocity,
                                       reference_linear_velocity=reference_linear_velocity,
                                       reference_angular_velocity=reference_angular_velocity,
                                       weight=weight)

        if check:
            full_root_link, full_tip_link = self.get_root_and_tip_link(root_link=root_link, root_group=root_group,
                                                                       tip_link=tip_link, tip_group=tip_group)
            goal_point = PointStamped()
            goal_point.header = goal_pose.header
            goal_point.point = goal_pose.pose.position
            self.add_goal_check(TranslationGoalChecker(self, full_tip_link, full_root_link, goal_point))
            goal_orientation = QuaternionStamped()
            goal_orientation.header = goal_pose.header
            goal_orientation.quaternion = goal_pose.pose.orientation
            self.add_goal_check(RotationGoalChecker(self, full_tip_link, full_root_link, goal_orientation))

    #
    # GENERAL GOAL STUFF ###############################################################################################
    #

    def plan_and_execute(self, expected_error_codes: List[int] = None, stop_after: float = None,
                         wait: bool = True) -> MoveResult:
        return self.send_goal(expected_error_codes=expected_error_codes, stop_after=stop_after, wait=wait)

    def plan(self, expected_error_codes: List[int] = None, wait: bool = True) -> MoveResult:
        return self.send_goal(expected_error_codes=expected_error_codes,
                              goal_type=MoveGoal.PLAN_ONLY,
                              wait=wait)

    def send_goal(self,
                  expected_error_codes: Optional[List[int]] = None,
                  goal_type: int = MoveGoal.PLAN_AND_EXECUTE,
                  goal: Optional[MoveGoal] = None,
                  stop_after: Optional[float] = None,
                  wait: bool = True) -> Optional[MoveResult]:
        try:
            time_spend_giskarding = time()
            if stop_after is not None:
                super().send_goal(goal_type, wait=False)
                rospy.sleep(stop_after)
                self.interrupt()
                rospy.sleep(1)
                r = self.get_result(rospy.Duration(3))
            elif not wait:
                super().send_goal(goal_type, wait=wait)
                return
            else:
                r = super().send_goal(goal_type, wait=wait)
            self.wait_heartbeats()
            diff = time() - time_spend_giskarding
            self.total_time_spend_giskarding += diff
            self.total_time_spend_moving += len(self.god_map.get_data(identifier.trajectory).keys()) * \
                                            self.god_map.get_data(identifier.sample_period)
            logging.logwarn(f'Goal processing took {diff}')
            error_code = MoveResult.SUCCESS
            cmd_id = 0
            for cmd_id in range(len(r.error_codes)):
                error_code = r.error_codes[cmd_id]
                error_message = r.error_messages[cmd_id]
                if expected_error_codes is None:
                    expected_error_code = MoveResult.SUCCESS
                else:
                    expected_error_code = expected_error_codes[cmd_id]
                assert error_code == expected_error_code, \
                    f'in goal {cmd_id}; ' \
                    f'got: {move_result_error_code(error_code)}, ' \
                    f'expected: {move_result_error_code(expected_error_code)} | error_massage: {error_message}'
            if error_code == MoveResult.SUCCESS:
                try:
                    self.wait_heartbeats(30)
                    for goal_checker in self.goal_checks[len(r.error_codes) - 1]:
                        goal_checker()
                except:
                    logging.logerr(f'Goal #{cmd_id} did\'t pass test.')
                    raise
            # self.are_joint_limits_violated()
        finally:
            self.goal_checks = defaultdict(list)
            self.sync_world_with_trajectory()
        return r

    def sync_world_with_trajectory(self):
        t = self.god_map.get_data(identifier.trajectory)
        whole_last_joint_state = t.get_last().to_position_dict()
        for group_name in self._object_js_topics:
            group_joints = self.get_group_info(group_name).joint_state.name
            group_last_joint_state = {str(k): v for k, v in whole_last_joint_state.items() if k in group_joints}
            self.set_object_joint_state(group_name, group_last_joint_state)

    def get_result_trajectory_position(self):
        trajectory = self.god_map.unsafe_get_data(identifier.trajectory)
        trajectory2 = {}
        for joint_name in trajectory.get_exact(0).keys():
            trajectory2[joint_name] = np.array([p[joint_name].position for t, p in trajectory.items()])
        return trajectory2

    def get_result_trajectory_velocity(self):
        trajectory = self.god_map.get_data(identifier.trajectory)
        trajectory2 = {}
        for joint_name in trajectory.get_exact(0).keys():
            trajectory2[joint_name] = np.array([p[joint_name].velocity for t, p in trajectory.items()])
        return trajectory2

    def are_joint_limits_violated(self, eps=1e-6):
        active_free_variables: List[FreeVariable] = self.god_map.get_data(identifier.qp_controller).free_variables
        for free_variable in active_free_variables:
            if free_variable.has_position_limits():
                lower_limit = free_variable.get_lower_limit(Derivatives.position)
                upper_limit = free_variable.get_upper_limit(Derivatives.position)
                if not isinstance(lower_limit, float):
                    lower_limit = lower_limit.evaluate()
                if not isinstance(upper_limit, float):
                    upper_limit = upper_limit.evaluate()
                current_position = self.world.state[free_variable.name].position
                assert lower_limit - eps <= current_position <= upper_limit + eps, \
                    f'joint limit of {free_variable.name} is violated {lower_limit} <= {current_position} <= {upper_limit}'

    def are_joint_limits_in_traj_violated(self):
        trajectory_vel = self.get_result_trajectory_velocity()
        trajectory_pos = self.get_result_trajectory_position()
        controlled_joints = self.god_map.get_data(identifier.controlled_joints)
        for joint_name in controlled_joints:
            if isinstance(self.world.joints[joint_name], OneDofJoint):
                if not self.world.is_joint_continuous(joint_name):
                    joint_limits = self.world.get_joint_position_limits(joint_name)
                    error_msg = f'{joint_name} has violated joint position limit'
                    eps = 0.0001
                    np.testing.assert_array_less(trajectory_pos[joint_name], joint_limits[1] + eps, error_msg)
                    np.testing.assert_array_less(-trajectory_pos[joint_name], -joint_limits[0] + eps, error_msg)
                vel_limit = self.world.get_joint_velocity_limits(joint_name)[1] * 1.001
                vel = trajectory_vel[joint_name]
                error_msg = f'{joint_name} has violated joint velocity limit {vel} > {vel_limit}'
                assert np.all(np.less_equal(vel, vel_limit)), error_msg
                assert np.all(np.greater_equal(vel, -vel_limit)), error_msg

    #
    # BULLET WORLD #####################################################################################################
    #

    TimeOut = 5000

    def register_group(self, new_group_name: str, root_link_group_name: str, root_link_name: str):
        super().register_group(new_group_name=new_group_name,
                               root_link_group_name=root_link_group_name,
                               root_link_name=root_link_name)
        assert new_group_name in self.get_group_names()

    def clear_world(self, timeout: float = TimeOut) -> UpdateWorldResponse:
        respone = super().clear_world(timeout=timeout)
        assert respone.error_codes == UpdateWorldResponse.SUCCESS
        assert len(self.world.groups) == 1
        assert len(self.get_group_names()) == 1
        assert self.original_number_of_links == len(self.world.links)
        return respone

    def remove_group(self,
                     name: str,
                     timeout: float = TimeOut,
                     expected_response: int = UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        old_link_names = []
        old_joint_names = []
        if expected_response == UpdateWorldResponse.SUCCESS:
            old_link_names = self.world.groups[name].link_names_as_set
            old_joint_names = self.world.groups[name].joint_names
        r = super(GiskardTestWrapper, self).remove_group(name, timeout=timeout)
        assert r.error_codes == expected_response, \
            f'Got: \'{update_world_error_code(r.error_codes)}\', ' \
            f'expected: \'{update_world_error_code(expected_response)}.\''
        assert name not in self.world.groups
        assert name not in self.get_group_names()
        if expected_response == UpdateWorldResponse.SUCCESS:
            for old_link_name in old_link_names:
                assert old_link_name not in self.world.link_names_as_set
            for old_joint_name in old_joint_names:
                assert old_joint_name not in self.world.joint_names
        return r

    def detach_group(self, name, timeout: float = TimeOut, expected_response=UpdateWorldResponse.SUCCESS):
        if expected_response == UpdateWorldResponse.SUCCESS:
            response = super().detach_group(name, timeout=timeout)
            self.check_add_object_result(response=response,
                                         name=name,
                                         size=None,
                                         pose=None,
                                         parent_link=self.world.root_link_name,
                                         parent_link_group='',
                                         expected_error_code=expected_response)

    def check_add_object_result(self,
                                response: UpdateWorldResponse,
                                name: str,
                                size: Optional,
                                pose: Optional[PoseStamped],
                                parent_link: str,
                                parent_link_group: str,
                                expected_error_code: int):
        assert response.error_codes == expected_error_code, \
            f'Got: \'{update_world_error_code(response.error_codes)}\', ' \
            f'expected: \'{update_world_error_code(expected_error_code)}.\''
        if expected_error_code == UpdateWorldResponse.SUCCESS:
            assert name in self.get_group_names()
            response2 = self.get_group_info(name)
            if pose is not None:
                p = self.transform_msg(self.world.root_link_name, pose)
                o_p = self.world.groups[name].base_pose
                compare_poses(p.pose, o_p)
                compare_poses(o_p, response2.root_link_pose.pose)
            if parent_link_group != '':
                robot = self.get_group_info(parent_link_group)
                assert name in robot.child_groups
                short_parent_link = self.world.groups[parent_link_group].get_link_short_name_match(parent_link)
                assert short_parent_link == self.world.get_parent_link_of_link(self.world.groups[name].root_link_name)
            else:
                if parent_link == '':
                    parent_link = self.world.root_link_name
                else:
                    parent_link = self.world.search_for_link_name(parent_link)
                assert parent_link == self.world.get_parent_link_of_link(self.world.groups[name].root_link_name)
        else:
            if expected_error_code != UpdateWorldResponse.DUPLICATE_GROUP_ERROR:
                assert name not in self.world.groups
                assert name not in self.get_group_names()

    def add_box(self,
                name: str,
                size: Tuple[float, float, float],
                pose: PoseStamped,
                parent_link: str = '',
                parent_link_group: str = '',
                timeout: float = TimeOut,
                expected_error_code: int = UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_box(name=name,
                                   size=size,
                                   pose=pose,
                                   parent_link=parent_link,
                                   parent_link_group=parent_link_group,
                                   timeout=timeout)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=size,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def update_group_pose(self, group_name: str, new_pose: PoseStamped, timeout: float = TimeOut,
                          expected_error_code=UpdateWorldResponse.SUCCESS):
        try:
            res = super().update_group_pose(group_name, new_pose, timeout)
        except UnknownGroupException as e:
            assert expected_error_code == UpdateWorldResponse.UNKNOWN_GROUP_ERROR
        if expected_error_code == UpdateWorldResponse.SUCCESS:
            info = self.get_group_info(group_name)
            map_T_group = tf.transform_pose(self.world.root_link_name, new_pose)
            compare_poses(info.root_link_pose.pose, map_T_group.pose)

    def add_sphere(self,
                   name: str,
                   radius: float = 1,
                   pose: PoseStamped = None,
                   parent_link: str = '',
                   parent_link_group: str = '',
                   timeout: float = TimeOut,
                   expected_error_code=UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_sphere(name=name,
                                      radius=radius,
                                      pose=pose,
                                      parent_link=parent_link,
                                      parent_link_group=parent_link_group,
                                      timeout=timeout)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def add_cylinder(self,
                     name: str,
                     height: float,
                     radius: float,
                     pose: PoseStamped = None,
                     parent_link: str = '',
                     parent_link_group: str = '',
                     timeout: float = TimeOut,
                     expected_error_code=UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_cylinder(name=name,
                                        height=height,
                                        radius=radius,
                                        pose=pose,
                                        parent_link=parent_link,
                                        parent_link_group=parent_link_group,
                                        timeout=timeout)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def add_mesh(self,
                 name: str = 'meshy',
                 mesh: str = '',
                 pose: PoseStamped = None,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 scale: Tuple[float, float, float] = (1, 1, 1),
                 timeout: float = TimeOut,
                 expected_error_code=UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_mesh(name=name,
                                    mesh=mesh,
                                    pose=pose,
                                    parent_link=parent_link,
                                    parent_link_group=parent_link_group,
                                    scale=scale,
                                    timeout=timeout)
        pose = utils.make_pose_from_parts(pose=pose, frame_id=pose.header.frame_id,
                                          position=pose.pose.position, orientation=pose.pose.orientation)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def add_urdf(self,
                 name: str,
                 urdf: str,
                 pose: PoseStamped,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 js_topic: Optional[str] = '',
                 set_js_topic: Optional[str] = '',
                 timeout: float = TimeOut,
                 expected_error_code=UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        response = super().add_urdf(name=name,
                                    urdf=urdf,
                                    pose=pose,
                                    parent_link=parent_link,
                                    parent_link_group=parent_link_group,
                                    js_topic=js_topic,
                                    set_js_topic=set_js_topic,
                                    timeout=timeout)
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def update_parent_link_of_group(self,
                                    name: str,
                                    parent_link: str = '',
                                    parent_link_group: str = '',
                                    timeout: float = TimeOut,
                                    expected_response: int = UpdateWorldResponse.SUCCESS) -> UpdateWorldResponse:
        r = super(GiskardTestWrapper, self).update_parent_link_of_group(name=name,
                                                                        parent_link=parent_link,
                                                                        parent_link_group=parent_link_group,
                                                                        timeout=timeout)
        self.wait_heartbeats()
        assert r.error_codes == expected_response, \
            f'Got: \'{update_world_error_code(r.error_codes)}\', ' \
            f'expected: \'{update_world_error_code(expected_response)}.\''
        if r.error_codes == r.SUCCESS:
            self.check_add_object_result(response=r,
                                         name=name,
                                         size=None,
                                         pose=None,
                                         parent_link=parent_link,
                                         parent_link_group=parent_link_group,
                                         expected_error_code=expected_response)
        return r

    def get_external_collisions(self) -> Collisions:
        collision_goals = []
        for robot_name in self.robot_names:
            collision_goals.append(CollisionEntry(type=CollisionEntry.AVOID_COLLISION,
                                                  distance=-1,
                                                  group1=robot_name))
            collision_goals.append(CollisionEntry(type=CollisionEntry.ALLOW_COLLISION,
                                                  distance=-1,
                                                  group1=robot_name,
                                                  group2=robot_name))
        return self.compute_collisions(collision_goals)

    def get_self_collisions(self, group_name: Optional[str] = None) -> Collisions:
        if group_name is None:
            group_name = self.robot_names[0]
        collision_entries = [CollisionEntry(type=CollisionEntry.AVOID_COLLISION,
                                            distance=-1,
                                            group1=group_name,
                                            group2=group_name)]
        return self.compute_collisions(collision_entries)

    def compute_collisions(self, collision_entries: List[CollisionEntry]) -> Collisions:
        self.collision_scene.reset_cache()
        collision_matrix = self.collision_scene.collision_goals_to_collision_matrix(collision_entries,
                                                                                    defaultdict(lambda: 0.3))

        return self.collision_scene.check_collisions(collision_matrix, 15)

    def compute_all_collisions(self) -> Collisions:
        collision_entries = [CollisionEntry(type=CollisionEntry.AVOID_COLLISION,
                                            distance=-1)]
        return self.compute_collisions(collision_entries)

    def check_cpi_geq(self, links, distance_threshold, check_external=True, check_self=True):
        collisions = self.compute_all_collisions()
        links = [self.world.search_for_link_name(link_name) for link_name in links]
        for collision in collisions.all_collisions:
            if not check_external and collision.is_external:
                continue
            if not check_self and not collision.is_external:
                continue
            if collision.original_link_a in links or collision.original_link_b in links:
                assert collision.contact_distance >= distance_threshold, \
                    f'{collision.contact_distance} < {distance_threshold} ' \
                    f'({collision.original_link_a} with {collision.original_link_b})'

    def check_cpi_leq(self, links, distance_threshold, check_external=True, check_self=True):
        collisions = self.compute_all_collisions()
        min_contact: Collision = None
        links = [self.world.search_for_link_name(link_name) for link_name in links]
        for collision in collisions.all_collisions:
            if not check_external and collision.is_external:
                continue
            if not check_self and not collision.is_external:
                continue
            if collision.original_link_a in links or collision.original_link_b in links:
                if min_contact is None or collision.contact_distance <= min_contact.contact_distance:
                    min_contact = collision
        assert min_contact.contact_distance <= distance_threshold, \
            f'{min_contact.contact_distance} > {distance_threshold} ' \
            f'({min_contact.original_link_a} with {min_contact.original_link_b})'

    def move_base(self, goal_pose: PoseStamped):
        pass

    def reset(self):
        pass

    def reset_base(self):
        p = PoseStamped()
        p.header.frame_id = tf.get_tf_root()
        p.pose.orientation.w = 1
        self.set_localization(p)
        self.wait_heartbeats()
        self.teleport_base(p)


#
# class PR2AndDonbot(GiskardTestWrapper):
#     default_pose_pr2 = {'r_elbow_flex_joint': -0.15,
#                         'r_forearm_roll_joint': 0,
#                         'r_shoulder_lift_joint': 0,
#                         'r_shoulder_pan_joint': 0,
#                         'r_upper_arm_roll_joint': 0,
#                         'r_wrist_flex_joint': -0.10001,
#                         'r_wrist_roll_joint': 0,
#                         'l_elbow_flex_joint': -0.15,
#                         'l_forearm_roll_joint': 0,
#                         'l_shoulder_lift_joint': 0,
#                         'l_shoulder_pan_joint': 0,
#                         'l_upper_arm_roll_joint': 0,
#                         'l_wrist_flex_joint': -0.10001,
#                         'l_wrist_roll_joint': 0,
#                         'torso_lift_joint': 0.2,
#                         'head_pan_joint': 0,
#                         'head_tilt_joint': 0}
#
#     better_pose_pr2 = {'r_shoulder_pan_joint': -1.7125,
#                        'r_shoulder_lift_joint': -0.25672,
#                        'r_upper_arm_roll_joint': -1.46335,
#                        'r_elbow_flex_joint': -2.12,
#                        'r_forearm_roll_joint': 1.76632,
#                        'r_wrist_flex_joint': -0.10001,
#                        'r_wrist_roll_joint': 0.05106,
#                        'l_shoulder_pan_joint': 1.9652,
#                        'l_shoulder_lift_joint': - 0.26499,
#                        'l_upper_arm_roll_joint': 1.3837,
#                        'l_elbow_flex_joint': -2.12,
#                        'l_forearm_roll_joint': 16.99,
#                        'l_wrist_flex_joint': - 0.10001,
#                        'l_wrist_roll_joint': 0,
#                        'torso_lift_joint': 0.2,
#
#                        'head_pan_joint': 0,
#                        'head_tilt_joint': 0,
#                        }
#
#     default_pose_donbot = {
#         'ur5_elbow_joint': 0.0,
#         'ur5_shoulder_lift_joint': 0.0,
#         'ur5_shoulder_pan_joint': 0.0,
#         'ur5_wrist_1_joint': 0.0,
#         'ur5_wrist_2_joint': 0.0,
#         'ur5_wrist_3_joint': 0.0
#     }
#
#     better_pose_donbot = {
#         'ur5_shoulder_pan_joint': -np.pi / 2,
#         'ur5_shoulder_lift_joint': -2.44177755311,
#         'ur5_elbow_joint': 2.15026930371,
#         'ur5_wrist_1_joint': 0.291547812391,
#         'ur5_wrist_2_joint': np.pi / 2,
#         'ur5_wrist_3_joint': np.pi / 2
#     }
#
#     def __init__(self):
#         self.r_tips = dict()
#         self.l_tips = dict()
#         self.r_grippers = dict()
#         self.l_grippers = dict()
#         self.set_localization_srvs = dict()
#         self.set_bases = dict()
#         self.default_roots = dict()
#         self.tf_prefix = dict()
#         self.camera_tip = 'camera_link'
#         self.gripper_tip = 'gripper_tool_frame'
#         self.gripper_pub_topic = 'wsg_50_driver/goal_position'
#         self.camera_tips = dict()
#         self.gripper_tips = dict()
#         self.gripper_pubs = dict()
#         self.default_poses = dict()
#         self.better_poses = dict()
#         self.pr2 = 'pr2'
#         self.donbot = 'donbot'
#         robot_names = [self.pr2, self.donbot]
#         super(PR2AndDonbot, self).__init__(u'package://giskardpy/config/pr2_and_donbot.yaml',
#                                            robot_names=robot_names, namespaces=robot_names)
#         for robot_name in self.collision_scene.robot_names:
#             if self.pr2 == robot_name:
#                 self.r_tips[robot_name] = u'r_gripper_tool_frame'
#                 self.l_tips[robot_name] = u'l_gripper_tool_frame'
#                 self.tf_prefix[robot_name] = robot_name.replace('/', '')
#                 self.r_grippers[robot_name] = rospy.ServiceProxy(
#                     u'/{}/r_gripper_simulator/set_joint_states'.format(robot_name), SetJointState)
#                 self.l_grippers[robot_name] = rospy.ServiceProxy(
#                     u'/{}/l_gripper_simulator/set_joint_states'.format(robot_name), SetJointState)
#                 self.set_localization_srvs[robot_name] = rospy.ServiceProxy(
#                     '/{}/map_odom_transform_publisher/update_map_odom_transform'.format(robot_name),
#                     UpdateTransform)
#                 self.set_bases[robot_name] = rospy.ServiceProxy(
#                     '/{}/base_simulator/set_joint_states'.format(robot_name), SetJointState)
#                 self.default_roots[robot_name] = self.world.groups[robot_name].root_link_name
#                 self.default_poses[robot_name] = self.default_pose_pr2
#                 self.better_poses[robot_name] = self.better_pose_pr2
#             else:
#                 self.camera_tips[robot_name] = PrefixName(self.camera_tip, robot_name)
#                 self.gripper_tips[robot_name] = PrefixName(self.gripper_tip, robot_name)
#                 self.default_roots[robot_name] = self.world.groups[robot_name].root_link_name
#                 self.set_localization_srvs[robot_name] = rospy.ServiceProxy(
#                     '/{}/map_odom_transform_publisher/update_map_odom_transform'.format(robot_name),
#                     UpdateTransform)
#                 self.gripper_pubs[robot_name] = rospy.Publisher(
#                     '/{}'.format(str(PrefixName(self.gripper_pub_topic, robot_name))),
#                     PositionCmd, queue_size=10)
#                 self.default_poses[robot_name] = self.default_pose_donbot
#                 self.better_poses[robot_name] = self.better_pose_donbot
#
#     def set_localization(self, map_T_odom, robot_name):
#         """
#         :type map_T_odom: PoseStamped
#         """
#         req = UpdateTransformRequest()
#         req.transform.translation = map_T_odom.pose.position
#         req.transform.rotation = map_T_odom.pose.orientation
#         assert self.set_localization_srvs[robot_name](req).success
#         self.wait_heartbeats(10)
#         p2 = self.world.compute_fk_pose(self.world.root_link_name, self.world.groups[robot_name].root_link_name)
#         compare_poses(p2.pose, map_T_odom.pose)
#
#     def open_gripper(self, robot_name):
#         self.set_gripper(robot_name, 0.109)
#
#     def close_gripper(self, robot_name):
#         self.set_gripper(robot_name, 0)
#
#     def set_gripper(self, robot_name, width, gripper_joint='gripper_joint'):
#         """
#         :param width: goal width in m
#         :type width: float
#         """
#         width = max(0.0065, min(0.109, width))
#         goal = PositionCmd()
#         goal.pos = width * 1000
#         rospy.sleep(0.5)
#         self.gripper_pubs[robot_name].publish(goal)
#         rospy.sleep(0.5)
#         self.wait_heartbeats()
#         np.testing.assert_almost_equal(self.world.state[PrefixName(gripper_joint, robot_name)].position, width,
#                                        decimal=3)
#
#     def reset_base(self, robot_name):
#         p = PoseStamped()
#         p.header.frame_id = self.map
#         p.pose.orientation.w = 1
#         self.set_localization(p, robot_name)
#         self.wait_heartbeats()
#         self.move_base(p, robot_name)
#
#     def clear_world(self):
#         return_val = super(GiskardTestWrapper, self).clear_world()
#         assert return_val.error_codes == UpdateWorldResponse.SUCCESS
#         assert len(self.world.groups) == 2
#         assert len(self.world.robot_names) == 2
#         assert self.original_number_of_links == len(self.world._links)
#
#     def reset(self):
#         self.clear_world()
#         for robot_name in self.namespaces:
#             self.reset_base(robot_name)
#             self.open_gripper(robot_name)
#
#     def move_base(self, goal_pose, robot_name):
#         if robot_name == self.pr2:
#             self.teleport_base(goal_pose, robot_name)
#         else:
#             goal_pose = tf.transform_pose(str(self.default_roots[robot_name]), goal_pose)
#             js = {'odom_x_joint': goal_pose.pose.position.x,
#                   'odom_y_joint': goal_pose.pose.position.y,
#                   'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
#                                                                           goal_pose.pose.orientation.y,
#                                                                           goal_pose.pose.orientation.z,
#                                                                           goal_pose.pose.orientation.w]))[0]}
#             self.allow_all_collisions()
#             self.set_joint_goal(js, group_name=robot_name, decimal=1)
#             self.plan_and_execute()
#
#     def get_l_gripper_links(self):
#         if 'l_gripper' not in self.world.group_names:
#             self.world.register_group('l_gripper', 'l_wrist_roll_link')
#         return [str(x) for x in self.world.groups['l_gripper'].link_names_with_collisions]
#
#     def get_r_gripper_links(self):
#         if 'r_gripper' not in self.world.group_names:
#             self.world.register_group('r_gripper', 'r_wrist_roll_link')
#         return [str(x) for x in self.world.groups['r_gripper'].link_names_with_collisions]
#
#     def get_r_forearm_links(self):
#         return [u'r_wrist_flex_link', u'r_wrist_roll_link', u'r_forearm_roll_link', u'r_forearm_link',
#                 u'r_forearm_link']
#
#     def get_allow_l_gripper(self, body_b=u'box'):
#         links = self.get_l_gripper_links()
#         return [CollisionEntry(CollisionEntry.ALLOW_COLLISION, 0, [link], body_b, []) for link in links]
#
#     def get_l_gripper_collision_entries(self, body_b=u'box', distance=0, action=CollisionEntry.ALLOW_COLLISION):
#         links = self.get_l_gripper_links()
#         return [CollisionEntry(action, distance, [link], body_b, []) for link in links]
#
#     def open_r_gripper(self, robot_name):
#         sjs = SetJointStateRequest()
#         sjs.state.name = [u'r_gripper_l_finger_joint', u'r_gripper_r_finger_joint', u'r_gripper_l_finger_tip_joint',
#                           u'r_gripper_r_finger_tip_joint']
#         sjs.state.position = [0.54, 0.54, 0.54, 0.54]
#         sjs.state.velocity = [0, 0, 0, 0]
#         sjs.state.effort = [0, 0, 0, 0]
#         self.r_grippers[robot_name].call(sjs)
#
#     def close_r_gripper(self, robot_name):
#         sjs = SetJointStateRequest()
#         sjs.state.name = [u'r_gripper_l_finger_joint', u'r_gripper_r_finger_joint', u'r_gripper_l_finger_tip_joint',
#                           u'r_gripper_r_finger_tip_joint']
#         sjs.state.position = [0, 0, 0, 0]
#         sjs.state.velocity = [0, 0, 0, 0]
#         sjs.state.effort = [0, 0, 0, 0]
#         self.r_grippers[robot_name].call(sjs)
#
#     def open_l_gripper(self, robot_name):
#         sjs = SetJointStateRequest()
#         sjs.state.name = [u'l_gripper_l_finger_joint', u'l_gripper_r_finger_joint', u'l_gripper_l_finger_tip_joint',
#                           u'l_gripper_r_finger_tip_joint']
#         sjs.state.position = [0.54, 0.54, 0.54, 0.54]
#         sjs.state.velocity = [0, 0, 0, 0]
#         sjs.state.effort = [0, 0, 0, 0]
#         self.l_grippers[robot_name].call(sjs)
#
#     def close_l_gripper(self, robot_name):
#         sjs = SetJointStateRequest()
#         sjs.state.name = [u'l_gripper_l_finger_joint', u'l_gripper_r_finger_joint', u'l_gripper_l_finger_tip_joint',
#                           u'l_gripper_r_finger_tip_joint']
#         sjs.state.position = [0, 0, 0, 0]
#         sjs.state.velocity = [0, 0, 0, 0]
#         sjs.state.effort = [0, 0, 0, 0]
#         self.l_grippers[robot_name].call(sjs)
#
#     def teleport_base(self, goal_pose, robot_name):
#         goal_pose = tf.transform_pose(str(self.default_roots[robot_name]), goal_pose)
#         js = {'odom_x_joint': goal_pose.pose.position.x,
#               'odom_y_joint': goal_pose.pose.position.y,
#               'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
#                                                                       goal_pose.pose.orientation.y,
#                                                                       goal_pose.pose.orientation.z,
#                                                                       goal_pose.pose.orientation.w]))[0]}
#         goal = SetJointStateRequest()
#         goal.state = position_dict_to_joint_states(js)
#         self.set_bases[robot_name].call(goal)
#         rospy.sleep(0.5)


# class Donbot2(GiskardTestWrapper):
#     default_pose = {
#         'ur5_elbow_joint': 0.0,
#         'ur5_shoulder_lift_joint': 0.0,
#         'ur5_shoulder_pan_joint': 0.0,
#         'ur5_wrist_1_joint': 0.0,
#         'ur5_wrist_2_joint': 0.0,
#         'ur5_wrist_3_joint': 0.0
#     }
#
#     better_pose = {
#         'ur5_shoulder_pan_joint': -np.pi / 2,
#         'ur5_shoulder_lift_joint': -2.44177755311,
#         'ur5_elbow_joint': 2.15026930371,
#         'ur5_wrist_1_joint': 0.291547812391,
#         'ur5_wrist_2_joint': np.pi / 2,
#         'ur5_wrist_3_joint': np.pi / 2
#     }
#
#     def __init__(self):
#         self.camera_tip = 'camera_link'
#         self.gripper_tip = 'gripper_tool_frame'
#         self.gripper_pub_topic = 'wsg_50_driver/goal_position'
#         self.camera_tips = dict()
#         self.gripper_tips = dict()
#         self.gripper_pubs = dict()
#         self.default_roots = dict()
#         self.set_localization_srvs = dict()
#         self.robot_names = ['donbot_a', 'donbot_b']
#         super(Donbot2, self).__init__('package://giskardpy/config/donbot_twice.yaml', self.robot_names,
#                                       self.robot_names)
#         for robot_name in self.namespaces:
#             self.camera_tips[robot_name] = PrefixName(self.camera_tip, robot_name)
#             self.gripper_tips[robot_name] = PrefixName(self.gripper_tip, robot_name)
#             self.default_roots[robot_name] = self.world.groups[robot_name].root_link_name
#             self.set_localization_srvs[robot_name] = rospy.ServiceProxy(
#                 '/{}/map_odom_transform_publisher/update_map_odom_transform'.format(robot_name),
#                 UpdateTransform)
#             self.gripper_pubs[robot_name] = rospy.Publisher(
#                 '/{}'.format(str(PrefixName(self.gripper_pub_topic, robot_name))),
#                 PositionCmd, queue_size=10)
#
#     def move_base(self, robot_name, goal_pose):
#         goal_pose = tf.transform_pose(str(self.default_roots[robot_name]), goal_pose)
#         js = {'odom_x_joint': goal_pose.pose.position.x,
#               'odom_y_joint': goal_pose.pose.position.y,
#               'odom_z_joint': rotation_from_matrix(quaternion_matrix([goal_pose.pose.orientation.x,
#                                                                       goal_pose.pose.orientation.y,
#                                                                       goal_pose.pose.orientation.z,
#                                                                       goal_pose.pose.orientation.w]))[0]}
#         self.allow_all_collisions()
#         self.set_joint_goal(js, group_name=robot_name, decimal=1)
#         self.plan_and_execute()
#
#     def set_localization(self, map_T_odom, robot_name):
#         """
#         :type map_T_odom: PoseStamped
#         """
#         req = UpdateTransformRequest()
#         req.transform.translation = map_T_odom.pose.position
#         req.transform.rotation = map_T_odom.pose.orientation
#         assert self.set_localization_srvs[robot_name](req).success
#         self.wait_heartbeats(10)
#         p2 = self.world.compute_fk_pose(self.world.root_link_name, self.world.groups[robot_name].root_link_name)
#         compare_poses(p2.pose, map_T_odom.pose)
#
#     def open_gripper(self, robot_name):
#         self.set_gripper(robot_name, 0.109)
#
#     def close_gripper(self, robot_name):
#         self.set_gripper(robot_name, 0)
#
#     def set_gripper(self, robot_name, width, gripper_joint='gripper_joint'):
#         """
#         :param width: goal width in m
#         :type width: float
#         """
#         width = max(0.0065, min(0.109, width))
#         goal = PositionCmd()
#         goal.pos = width * 1000
#         self.gripper_pubs[robot_name].publish(goal)
#         rospy.sleep(0.5)
#         self.wait_heartbeats()
#         np.testing.assert_almost_equal(
#             self.world.groups[robot_name].state[str(PrefixName(gripper_joint, robot_name))].position, width, decimal=3)
#
#     def reset_base(self, robot_name):
#         p = PoseStamped()
#         p.header.frame_id = self.map
#         p.pose.orientation.w = 1
#         self.set_localization(p, robot_name)
#         self.wait_heartbeats()
#         self.move_base(robot_name, p)
#
#     def clear_world(self):
#         return_val = super(GiskardTestWrapper, self).clear_world()
#         assert return_val.error_codes == UpdateWorldResponse.SUCCESS
#         assert len(self.world.groups) == 2
#         assert len(self.robot_names) == 2
#         assert self.original_number_of_links == len(self.world._links)
#
#     def reset(self):
#         self.clear_world()
#         for robot_name in self.namespaces:
#             self.reset_base(robot_name)
#             self.open_gripper(robot_name)


def publish_marker_sphere(position, frame_id='map', radius=0.05, id_=0):
    m = Marker()
    m.action = m.ADD
    m.ns = 'debug'
    m.id = id_
    m.type = m.SPHERE
    m.header.frame_id = frame_id
    m.pose.position.x = position[0]
    m.pose.position.y = position[1]
    m.pose.position.z = position[2]
    m.color = ColorRGBA(1, 0, 0, 1)
    m.scale.x = radius
    m.scale.y = radius
    m.scale.z = radius

    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    start = rospy.get_rostime()
    while pub.get_num_connections() < 1 and (rospy.get_rostime() - start).to_sec() < 2:
        # wait for a connection to publisher
        # you can do whatever you like here or simply do nothing
        pass

    pub.publish(m)


def publish_marker_vector(start: Point, end: Point, diameter_shaft: float = 0.01, diameter_head: float = 0.02,
                          id_: int = 0):
    """
    assumes points to be in frame map
    """
    m = Marker()
    m.action = m.ADD
    m.ns = 'debug'
    m.id = id_
    m.type = m.ARROW
    m.points.append(start)
    m.points.append(end)
    m.color = ColorRGBA(1, 0, 0, 1)
    m.scale.x = diameter_shaft
    m.scale.y = diameter_head
    m.scale.z = 0
    m.header.frame_id = 'map'

    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    start = rospy.get_rostime()
    while pub.get_num_connections() < 1 and (rospy.get_rostime() - start).to_sec() < 2:
        # wait for a connection to publisher
        # you can do whatever you like here or simply do nothing
        pass
    rospy.sleep(0.3)

    pub.publish(m)


class SuccessfulActionServer(object):
    def __init__(self):
        self.name_space = rospy.get_param('~name_space')
        self.joint_names = rospy.get_param('~joint_names')
        self.state = {j: 0 for j in self.joint_names}
        self.pub = rospy.Publisher('{}/state'.format(self.name_space), control_msgs.msg.JointTrajectoryControllerState,
                                   queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.state_cb)
        self._as = actionlib.SimpleActionServer(self.name_space, control_msgs.msg.FollowJointTrajectoryAction,
                                                execute_cb=self.execute_cb, auto_start=False)
        self._as.start()
        self._as.register_preempt_callback(self.preempt_requested)

    def state_cb(self, timer_event):
        msg = control_msgs.msg.JointTrajectoryControllerState()
        msg.header.stamp = timer_event.current_real
        msg.joint_names = self.joint_names
        self.pub.publish(msg)

    def preempt_requested(self):
        print('cancel called')
        self._as.set_preempted()

    def execute_cb(self, goal):
        rospy.sleep(goal.trajectory.points[-1].time_from_start)
        if self._as.is_active():
            self._as.set_succeeded()


class GoalChecker:
    def __init__(self, giskard: GiskardTestWrapper):
        self.giskard = giskard
        self.god_map = giskard.god_map
        self.world: WorldTree = self.god_map.unsafe_get_data(identifier.world)
        # self.robot: SubWorldTree = self.world.groups[self.god_map.unsafe_get_data(identifier.robot_group_name)]


class JointGoalChecker(GoalChecker):
    def __init__(self, giskard, goal_state, decimal=2):
        super().__init__(giskard)
        self.goal_state = goal_state
        self.decimal = decimal

    def get_current_joint_state(self) -> JointStates:
        return self.world.state

    def __call__(self):
        current_joint_state = self.get_current_joint_state()
        self.compare_joint_state(current_joint_state, self.goal_state, decimal=self.decimal)

    def compare_joint_state(self, current_js, goal_js, decimal=2):
        """
        :type current_js: dict
        :type goal_js: dict
        :type decimal: int
        """
        for joint_name in goal_js:
            full_joint_name = self.world.search_for_joint_name(joint_name)
            goal = goal_js[joint_name]
            current = current_js[full_joint_name].position
            if self.world.is_joint_continuous(full_joint_name):
                np.testing.assert_almost_equal(shortest_angular_distance(goal, current), 0, decimal=decimal,
                                               err_msg='{}: actual: {} desired: {}'.format(full_joint_name, current,
                                                                                           goal))
            else:
                np.testing.assert_almost_equal(current, goal, decimal,
                                               err_msg='{}: actual: {} desired: {}'.format(full_joint_name, current,
                                                                                           goal))


class TranslationGoalChecker(GoalChecker):
    def __init__(self, giskard, tip_link, root_link, expected):
        super().__init__(giskard)
        self.expected = deepcopy(expected)
        self.tip_link = tip_link
        self.root_link = root_link
        self.expected = self.giskard.transform_msg(self.root_link, self.expected)

    def __call__(self):
        expected = self.expected
        current_pose = self.world.compute_fk_pose(self.root_link, self.tip_link)
        np.testing.assert_array_almost_equal(msg_to_list(expected.point),
                                             msg_to_list(current_pose.pose.position), decimal=2)


class AlignPlanesGoalChecker(GoalChecker):
    def __init__(self, giskard, tip_link, tip_normal, root_link, root_normal):
        super().__init__(giskard)
        self.tip_normal = tip_normal
        self.tip_link = tip_link
        self.root_link = root_link
        self.expected = self.giskard.transform_msg(self.root_link, root_normal)

    def __call__(self):
        expected = self.expected
        current = self.giskard.transform_msg(self.root_link, self.tip_normal)
        np.testing.assert_array_almost_equal(msg_to_list(expected.vector), msg_to_list(current.vector), decimal=2)


class PointingGoalChecker(GoalChecker):
    def __init__(self, giskard, tip_link, goal_point, root_link, pointing_axis=None):
        super().__init__(giskard)
        self.tip_link = tip_link
        self.root_link = root_link
        if pointing_axis:
            self.tip_V_pointer = self.giskard.transform_msg(self.tip_link, pointing_axis)
        else:
            self.tip_V_pointer = Vector3Stamped()
            self.tip_V_pointer.vector.z = 1
            self.tip_V_pointer.header.frame_id = tip_link
        self.tip_V_pointer = tf.msg_to_homogeneous_matrix(self.tip_V_pointer)
        self.goal_point = self.giskard.transform_msg(self.root_link, goal_point)
        self.goal_point.header.stamp = rospy.Time()

    def __call__(self):
        tip_P_goal = tf.msg_to_homogeneous_matrix(self.giskard.transform_msg(self.tip_link, self.goal_point))
        tip_P_goal[-1] = 0
        tip_P_goal = tip_P_goal / np.linalg.norm(tip_P_goal)
        np.testing.assert_array_almost_equal(tip_P_goal, self.tip_V_pointer, decimal=2)


class RotationGoalChecker(GoalChecker):
    def __init__(self, giskard, tip_link, root_link, expected: QuaternionStamped):
        super().__init__(giskard)
        self.expected = deepcopy(expected)
        self.tip_link = tip_link
        self.root_link = root_link
        self.expected = self.giskard.transform_msg(self.root_link, self.expected)

    def __call__(self):
        expected = self.expected
        current_pose = self.world.compute_fk_pose(self.root_link, self.tip_link)

        try:
            np.testing.assert_array_almost_equal(msg_to_list(expected.quaternion),
                                                 msg_to_list(current_pose.pose.orientation), decimal=2)
        except AssertionError:
            np.testing.assert_array_almost_equal(msg_to_list(expected.quaternion),
                                                 -np.array(msg_to_list(current_pose.pose.orientation)), decimal=2)
