import csv
import keyword
import os
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Queue
from time import time
from typing import Tuple, Optional, List, Dict, Union

import hypothesis.strategies as st
import numpy as np
import rospy
from angles import shortest_angular_distance
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from hypothesis import assume
from hypothesis.strategies import composite
from numpy import pi
from rospy import Timer
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_py import LookupException, ExtrapolationException
from visualization_msgs.msg import Marker

import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import CollisionEntry, MoveResult, MoveGoal, WorldResult, GiskardError
from giskard_msgs.srv import DyeGroupResponse
from giskardpy.configs.giskard import Giskard
from giskardpy.configs.qp_controller_config import SupportedQPSolver
from giskardpy.data_types import KeyDefaultDict
from giskardpy.exceptions import UnknownGroupException
from giskardpy.goals.diff_drive_goals import DiffDriveTangentialToPoint, KeepHandInWorkspace
from giskardpy.god_map import god_map
from giskardpy.model.collision_world_syncer import Collisions, Collision
from giskardpy.model.joints import OneDofJoint, OmniDrive, DiffDrive
from giskardpy.data_types import PrefixName, Derivatives
from giskardpy.python_interface.old_python_interface import OldGiskardWrapper
from giskardpy.qp.free_variable import FreeVariable
from giskardpy.qp.qp_controller import available_solvers
from giskardpy.tasks.task import WEIGHT_ABOVE_CA
from giskardpy.utils import logging, utils
from giskardpy.utils.math import compare_poses
from giskardpy.utils.utils import resolve_ros_iris, position_dict_to_joint_states

BIG_NUMBER = 1e100
SMALL_NUMBER = 1e-100

folder_name = 'tmp_data/'


def vector(x):
    return st.lists(float_no_nan_no_inf(), min_size=x, max_size=x)


error_codes = {value: name for name, value in vars(GiskardError).items() if
               isinstance(value, int) and name[0].isupper()}


def error_code_to_name(code):
    return error_codes[code]


def angle_positive():
    return st.floats(0, 2 * np.pi)


def random_angle():
    return st.floats(-np.pi, np.pi)


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


class GiskardTestWrapper(OldGiskardWrapper):
    default_pose = {}
    better_pose = {}
    odom_root = 'odom'

    def __init__(self,
                 giskard: Giskard):
        self.total_time_spend_giskarding = 0
        self.total_time_spend_moving = 0
        self._alive = True
        self.default_env_name: Optional[str] = None
        self.env_joint_state_pubs: Dict[str, rospy.Publisher] = {}

        try:
            from iai_naive_kinematics_sim.srv import UpdateTransform
            self.set_localization_srv = rospy.ServiceProxy('/map_odom_transform_publisher/update_map_odom_transform',
                                                           UpdateTransform)
        except Exception as e:
            self.set_localization_srv = None

        self.giskard = giskard
        self.giskard.grow()
        if god_map.is_in_github_workflow():
            logging.loginfo('Inside github workflow, turning off visualization')
            god_map.tree.turn_off_visualization()
        if 'QP_SOLVER' in os.environ:
            god_map.qp_controller_config.set_qp_solver(SupportedQPSolver[os.environ['QP_SOLVER']])
        self.heart = Timer(period=rospy.Duration(god_map.tree.tick_rate), callback=self.heart_beat)
        # self.namespaces = namespaces
        self.robot_names = [list(god_map.world.groups.keys())[0]]
        super().__init__(node_name='tests')
        self.results = Queue(100)
        self.default_root = str(god_map.world.root_link_name)
        self.goal_checks = []

        def create_publisher(topic):
            p = rospy.Publisher(topic, JointState, queue_size=10)
            rospy.sleep(.2)
            return p

        self.joint_state_publisher = KeyDefaultDict(create_publisher)
        # rospy.sleep(1)
        self.original_number_of_links = len(god_map.world.links)

    def has_odometry_joint(self, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.robot_name
        parent_joint_name = god_map.world.groups[group_name].root_link.parent_joint_name
        if parent_joint_name is None:
            return False
        joint = god_map.world.get_joint(parent_joint_name)
        return isinstance(joint, (OmniDrive, DiffDrive))

    def set_seed_odometry(self, base_pose, group_name: Optional[str] = None):
        if group_name is None:
            group_name = self.robot_name
        self.motion_goals.set_seed_odometry(group_name=group_name,
                                            base_pose=base_pose)

    def set_localization(self, map_T_odom: PoseStamped):
        if self.set_localization_srv is not None:
            from iai_naive_kinematics_sim.srv import UpdateTransformRequest
            map_T_odom.pose.position.z = 0
            req = UpdateTransformRequest()
            req.transform.translation = map_T_odom.pose.position
            req.transform.rotation = map_T_odom.pose.orientation
            assert self.set_localization_srv(req).success
            self.wait_heartbeats(15)
            p2 = god_map.world.compute_fk_pose(god_map.world.root_link_name, self.odom_root)
            compare_poses(p2.pose, map_T_odom.pose)

    def transform_msg(self, target_frame, msg, timeout=1):
        result_msg = deepcopy(msg)
        try:
            if not god_map.is_standalone():
                return tf.transform_msg(target_frame, result_msg, timeout=timeout)
            else:
                raise LookupException('just to trigger except block')
        except (LookupException, ExtrapolationException) as e:
            target_frame = god_map.world.search_for_link_name(target_frame)
            try:
                result_msg.header.frame_id = god_map.world.search_for_link_name(result_msg.header.frame_id)
            except UnknownGroupException:
                pass
            return god_map.world.transform_msg(target_frame, result_msg)

    def wait_heartbeats(self, number=2):
        behavior_tree = god_map.tree
        c = behavior_tree.count
        while behavior_tree.count < c + number:
            rospy.sleep(0.001)

    def dye_group(self, group_name: str, rgba: Tuple[float, float, float, float],
                  expected_error_codes=(DyeGroupResponse.SUCCESS,)):
        res = self.world.dye_group(group_name, rgba)
        assert res.error_codes in expected_error_codes

    def heart_beat(self, timer_thing):
        if self._alive:
            god_map.tree.tick()

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
                                        str(int(god_map.qp_controller_config.max_derivative)),
                                        str(times)])

        logging.loginfo('saved benchmark file')

    def tear_down(self):
        self.print_qp_solver_times()
        rospy.sleep(1)
        self.heart.shutdown()
        # TODO it is strange that I need to kill the services... should be investigated. (:
        god_map.tree.kill_all_services()
        giskarding_time = self.total_time_spend_giskarding
        if not god_map.is_standalone():
            giskarding_time -= self.total_time_spend_moving
        logging.loginfo(f'total time spend giskarding: {giskarding_time}')
        logging.loginfo(f'total time spend moving: {self.total_time_spend_moving}')
        logging.loginfo('stopping tree')

    def set_env_state(self, joint_state: Dict[str, float], object_name: Optional[str] = None):
        if object_name is None:
            object_name = self.default_env_name
        if god_map.is_standalone():
            self.set_seed_configuration(joint_state)
            self.allow_all_collisions()
            self.plan_and_execute()
        else:
            joint_state_msg = position_dict_to_joint_states(joint_state)
            self.env_joint_state_pubs[object_name].publish(joint_state_msg)
        self.wait_heartbeats(3)
        current_js = god_map.world.groups[object_name].state
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

    def compare_joint_state(self, current_js: Dict[Union[str, PrefixName], float],
                            goal_js: Dict[Union[str, PrefixName], float],
                            decimal: int = 2):
        for joint_name in goal_js:
            goal = goal_js[joint_name]
            current = current_js[joint_name]
            if isinstance(joint_name, str):
                joint_name = god_map.world.search_for_joint_name(joint_name)
            if joint_name in god_map.world.joints and god_map.world.is_joint_continuous(joint_name):
                np.testing.assert_almost_equal(shortest_angular_distance(goal, current), 0, decimal=decimal,
                                               err_msg=f'{joint_name}: actual: {current} desired: {goal}')
            else:
                np.testing.assert_almost_equal(current, goal, decimal,
                                               err_msg=f'{joint_name}: actual: {current} desired: {goal}')

    #
    # GOAL STUFF #################################################################################################
    #

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

    def set_keep_hand_in_workspace(self, tip_link, map_frame=None, base_footprint=None):
        self.motion_goals.add_motion_goal(motion_goal_class=KeepHandInWorkspace.__name__,
                                          tip_link=tip_link,
                                          map_frame=map_frame,
                                          base_footprint=base_footprint)

    def set_diff_drive_tangential_to_point(self, goal_point: PointStamped, weight: float = WEIGHT_ABOVE_CA, **kwargs):
        self.motion_goals.add_motion_goal(motion_goal_class=DiffDriveTangentialToPoint.__name__,
                                          goal_point=goal_point,
                                          weight=weight,
                                          **kwargs)

    #
    # GENERAL GOAL STUFF ###############################################################################################
    #

    def execute(self, expected_error_code: int = GiskardError.SUCCESS, stop_after: float = None,
                wait: bool = True, add_local_minimum_reached: bool = True) -> MoveResult:
        if add_local_minimum_reached:
            self.add_default_end_motion_conditions()
        return self.send_goal(expected_error_code=expected_error_code, stop_after=stop_after, wait=wait)

    def projection(self, expected_error_code: int = GiskardError.SUCCESS, wait: bool = True,
                   add_local_minimum_reached: bool = True) -> MoveResult:
        """
        Plans, but doesn't execute the goal. Useful, if you just want to look at the planning ghost.
        :param wait: this function blocks if wait=True
        :return: result from Giskard
        """
        if add_local_minimum_reached:
            self.add_default_end_motion_conditions()
        last_js = god_map.world.state.to_position_dict()
        for key, value in list(last_js.items()):
            if key not in god_map.controlled_joints:
                del last_js[key]
        result = self.send_goal(expected_error_code=expected_error_code,
                                goal_type=MoveGoal.PROJECTION,
                                wait=wait)
        new_js = god_map.world.state.to_position_dict()
        for key, value in list(new_js.items()):
            if key not in god_map.controlled_joints:
                del new_js[key]
        self.compare_joint_state(new_js, last_js)
        return result

    def plan_and_execute(self, expected_error_code: int = GiskardError.SUCCESS, stop_after: float = None,
                         wait: bool = True) -> MoveResult:
        return self.execute(expected_error_code, stop_after, wait)

    def plan(self, expected_error_codes: int = GiskardError.SUCCESS, wait: bool = True,
             add_local_minimum_reached: bool = True) -> MoveResult:
        return self.projection(expected_error_code=expected_error_codes,
                               wait=wait,
                               add_local_minimum_reached=add_local_minimum_reached)

    def send_goal(self,
                  expected_error_code: int = GiskardError.SUCCESS,
                  goal_type: int = MoveGoal.EXECUTE,
                  goal: Optional[MoveGoal] = None,
                  stop_after: Optional[float] = None,
                  wait: bool = True) -> Optional[MoveResult]:
        try:
            time_spend_giskarding = time()
            if stop_after is not None:
                super()._send_action_goal(goal_type, wait=False)
                rospy.sleep(stop_after)
                self.interrupt()
                rospy.sleep(1)
                r = self.get_result(rospy.Duration(3))
            elif not wait:
                super()._send_action_goal(goal_type, wait=wait)
                return
            else:
                r = super()._send_action_goal(goal_type, wait=wait)
            self.wait_heartbeats()
            diff = time() - time_spend_giskarding
            self.total_time_spend_giskarding += diff
            self.total_time_spend_moving += (len(god_map.trajectory.keys()) *
                                             god_map.qp_controller_config.sample_period)
            logging.logwarn(f'Goal processing took {diff}')
            error_code = r.error.code
            error_message = r.error.msg
            assert error_code == expected_error_code, \
                f'got: {error_code_to_name(error_code)}, ' \
                f'expected: {error_code_to_name(expected_error_code)} | error_massage: {error_message}'
            if error_code == GiskardError.SUCCESS:
                try:
                    self.wait_heartbeats(30)
                    for goal_checker in self.goal_checks:
                        goal_checker()
                except:
                    logging.logerr(f'Goal did\'t pass test.')
                    raise
            # self.are_joint_limits_violated()
        finally:
            self.goal_checks = []
            self.sync_world_with_trajectory()
        return r

    def sync_world_with_trajectory(self):
        t = god_map.trajectory
        whole_last_joint_state = t.get_last().to_position_dict()
        for group_name in self.env_joint_state_pubs:
            group_joints = self.world.get_group_info(group_name).joint_state.name
            group_last_joint_state = {str(k): v for k, v in whole_last_joint_state.items() if k in group_joints}
            self.set_env_state(group_last_joint_state, group_name)

    def get_result_trajectory_position(self):
        trajectory = god_map.trajectory
        trajectory2 = {}
        for joint_name in trajectory.get_exact(0).keys():
            trajectory2[joint_name] = np.array([p[joint_name].position for t, p in trajectory.items()])
        return trajectory2

    def get_result_trajectory_velocity(self):
        trajectory = god_map.trajectory
        trajectory2 = {}
        for joint_name in trajectory.get_exact(0).keys():
            trajectory2[joint_name] = np.array([p[joint_name].velocity for t, p in trajectory.items()])
        return trajectory2

    def are_joint_limits_violated(self, eps=1e-2):
        active_free_variables: List[FreeVariable] = god_map.qp_controller.free_variables
        for free_variable in active_free_variables:
            if free_variable.has_position_limits():
                lower_limit = free_variable.get_lower_limit(Derivatives.position)
                upper_limit = free_variable.get_upper_limit(Derivatives.position)
                if not isinstance(lower_limit, float):
                    lower_limit = lower_limit.evaluate()
                if not isinstance(upper_limit, float):
                    upper_limit = upper_limit.evaluate()
                current_position = god_map.world.state[free_variable.name].position
                assert lower_limit - eps <= current_position <= upper_limit + eps, \
                    f'joint limit of {free_variable.name} is violated {lower_limit} <= {current_position} <= {upper_limit}'

    def are_joint_limits_in_traj_violated(self):
        trajectory_vel = self.get_result_trajectory_velocity()
        trajectory_pos = self.get_result_trajectory_position()
        controlled_joints = god_map.world.controlled_joints
        for joint_name in controlled_joints:
            if isinstance(god_map.world.joints[joint_name], OneDofJoint):
                if not god_map.world.is_joint_continuous(joint_name):
                    joint_limits = god_map.world.get_joint_position_limits(joint_name)
                    error_msg = f'{joint_name} has violated joint position limit'
                    eps = 0.0001
                    np.testing.assert_array_less(trajectory_pos[joint_name], joint_limits[1] + eps, error_msg)
                    np.testing.assert_array_less(-trajectory_pos[joint_name], -joint_limits[0] + eps, error_msg)
                vel_limit = god_map.world.get_joint_velocity_limits(joint_name)[1] * 1.001
                vel = trajectory_vel[joint_name]
                error_msg = f'{joint_name} has violated joint velocity limit {vel} > {vel_limit}'
                assert np.all(np.less_equal(vel, vel_limit)), error_msg
                assert np.all(np.greater_equal(vel, -vel_limit)), error_msg

    #
    # BULLET WORLD #####################################################################################################
    #

    TimeOut = 5000

    def register_group(self, new_group_name: str, root_link_group_name: str, root_link_name: str):
        self.world.register_group(new_group_name=new_group_name,
                                  root_link_group_name=root_link_group_name,
                                  root_link_name=root_link_name)
        self.wait_heartbeats()
        assert new_group_name in self.world.get_group_names()

    def clear_world(self) -> WorldResult:
        respone = self.world.clear()
        self.wait_heartbeats()
        self.default_env_name = None
        assert respone.error.code == GiskardError.SUCCESS
        assert len(god_map.world.groups) == 1
        assert len(self.world.get_group_names()) == 1
        assert self.original_number_of_links == len(god_map.world.links)
        return respone

    def remove_group(self,
                     name: str,
                     expected_response: int = GiskardError.SUCCESS) -> WorldResult:
        old_link_names = []
        old_joint_names = []
        if expected_response == GiskardError.SUCCESS:
            old_link_names = god_map.world.groups[name].link_names_as_set
            old_joint_names = god_map.world.groups[name].joint_names
        r = self.world.remove_group(name)
        self.wait_heartbeats()
        assert r.error.code == expected_response, \
            f'Got: \'{error_code_to_name(r.error.code)}\', ' \
            f'expected: \'{error_code_to_name(expected_response)}.\''
        assert name not in god_map.world.groups
        assert name not in self.world.get_group_names()
        if expected_response == GiskardError.SUCCESS:
            # links removed from world
            for old_link_name in old_link_names:
                assert old_link_name not in god_map.world.link_names_as_set
            # joints removed from world
            for old_joint_name in old_joint_names:
                assert old_joint_name not in god_map.world.joint_names
            # links removed from collision scene
            for link_a, link_b in god_map.collision_scene.self_collision_matrix:
                try:
                    assert link_a not in old_link_names
                    assert link_b not in old_link_names
                except AssertionError as e:
                    pass
        if name in self.env_joint_state_pubs:
            self.env_joint_state_pubs[name].unregister()
            del self.env_joint_state_pubs[name]
        if name == self.default_env_name:
            self.default_env_name = None
        return r

    def detach_group(self, name, expected_response=GiskardError.SUCCESS):
        if expected_response == GiskardError.SUCCESS:
            response = self.world.detach_group(name)
            self.wait_heartbeats()
            self.check_add_object_result(response=response,
                                         name=name,
                                         size=None,
                                         pose=None,
                                         parent_link=god_map.world.root_link_name,
                                         parent_link_group='',
                                         expected_error_code=expected_response)

    def check_add_object_result(self,
                                response: WorldResult,
                                name: str,
                                size: Optional,
                                pose: Optional[PoseStamped],
                                parent_link: str,
                                parent_link_group: str,
                                expected_error_code: int):
        assert response.error.code == expected_error_code, \
            f'Got: \'{error_code_to_name(response.error.code)}\', ' \
            f'expected: \'{error_code_to_name(expected_error_code)}.\''
        if expected_error_code == GiskardError.SUCCESS:
            assert name in self.world.get_group_names()
            response2 = self.world.get_group_info(name)
            if pose is not None:
                p = self.transform_msg(god_map.world.root_link_name, pose)
                o_p = god_map.world.groups[name].base_pose
                compare_poses(p.pose, o_p)
                compare_poses(o_p, response2.root_link_pose.pose)
            if parent_link_group != '':
                robot = self.world.get_group_info(parent_link_group)
                assert name in robot.child_groups
                short_parent_link = god_map.world.groups[parent_link_group].get_link_short_name_match(parent_link)
                assert short_parent_link == god_map.world.get_parent_link_of_link(
                    god_map.world.groups[name].root_link_name)
            else:
                if parent_link == '':
                    parent_link = god_map.world.root_link_name
                else:
                    parent_link = god_map.world.search_for_link_name(parent_link)
                if parent_link in god_map.world.robots[0].link_names:
                    object_links = god_map.world.groups[name].link_names
                    for link_a, link_b in god_map.collision_scene.self_collision_matrix:
                        if link_a in object_links or link_b in object_links:
                            break
                    else:
                        assert False, f'{name} not in collision matrix'
                assert parent_link == god_map.world.get_parent_link_of_link(god_map.world.groups[name].root_link_name)
        else:
            if expected_error_code != GiskardError.DUPLICATE_NAME:
                assert name not in god_map.world.groups
                assert name not in self.world.get_group_names()

    def add_box_to_world(self,
                         name: str,
                         size: Tuple[float, float, float],
                         pose: PoseStamped,
                         parent_link: str = '',
                         parent_link_group: str = '',
                         expected_error_code: int = GiskardError.SUCCESS) -> WorldResult:
        response = self.world.add_box(name=name,
                                      size=size,
                                      pose=pose,
                                      parent_link=parent_link,
                                      parent_link_group=parent_link_group)
        self.wait_heartbeats()
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=size,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def update_group_pose(self, group_name: str, new_pose: PoseStamped,
                          expected_error_code=GiskardError.SUCCESS):
        try:
            res = self.world.update_group_pose(group_name=group_name, new_pose=new_pose)
            self.wait_heartbeats()
        except UnknownGroupException as e:
            assert expected_error_code == GiskardError.UNKNOWN_GROUP
        if expected_error_code == GiskardError.SUCCESS:
            info = self.world.get_group_info(group_name)
            map_T_group = tf.transform_pose(god_map.world.root_link_name, new_pose)
            compare_poses(info.root_link_pose.pose, map_T_group.pose)

    def add_sphere_to_world(self,
                            name: str,
                            radius: float = 1,
                            pose: PoseStamped = None,
                            parent_link: str = '',
                            parent_link_group: str = '',
                            expected_error_code=GiskardError.SUCCESS) -> WorldResult:
        response = self.world.add_sphere(name=name,
                                         radius=radius,
                                         pose=pose,
                                         parent_link=parent_link,
                                         parent_link_group=parent_link_group)
        self.wait_heartbeats()
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def add_cylinder_to_world(self,
                              name: str,
                              height: float,
                              radius: float,
                              pose: PoseStamped = None,
                              parent_link: str = '',
                              parent_link_group: str = '',
                              expected_error_code=GiskardError.SUCCESS) -> WorldResult:
        response = self.world.add_cylinder(name=name,
                                           height=height,
                                           radius=radius,
                                           pose=pose,
                                           parent_link=parent_link,
                                           parent_link_group=parent_link_group)
        self.wait_heartbeats()
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        return response

    def add_mesh_to_world(self,
                          name: str = 'meshy',
                          mesh: str = '',
                          pose: PoseStamped = None,
                          parent_link: str = '',
                          parent_link_group: str = '',
                          scale: Tuple[float, float, float] = (1, 1, 1),
                          expected_error_code=GiskardError.SUCCESS) -> WorldResult:
        response = self.world.add_mesh(name=name,
                                       mesh=mesh,
                                       pose=pose,
                                       parent_link=parent_link,
                                       parent_link_group=parent_link_group,
                                       scale=scale)
        self.wait_heartbeats()
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

    def add_urdf_to_world(self,
                          name: str,
                          urdf: str,
                          pose: PoseStamped,
                          parent_link: str = '',
                          parent_link_group: str = '',
                          js_topic: Optional[str] = '',
                          set_js_topic: Optional[str] = '',
                          expected_error_code=GiskardError.SUCCESS) -> WorldResult:
        response = self.world.add_urdf(name=name,
                                       urdf=urdf,
                                       pose=pose,
                                       parent_link=parent_link,
                                       parent_link_group=parent_link_group,
                                       js_topic=js_topic)
        self.wait_heartbeats()
        self.check_add_object_result(response=response,
                                     name=name,
                                     size=None,
                                     pose=pose,
                                     parent_link=parent_link,
                                     parent_link_group=parent_link_group,
                                     expected_error_code=expected_error_code)
        if set_js_topic:
            self.env_joint_state_pubs[name] = rospy.Publisher(set_js_topic, JointState, queue_size=10)
        if self.default_env_name is None:
            self.default_env_name = name
        return response

    def update_parent_link_of_group(self,
                                    name: str,
                                    parent_link: str = '',
                                    parent_link_group: str = '',
                                    expected_response: int = GiskardError.SUCCESS) -> WorldResult:
        r = self.world.update_parent_link_of_group(name=name,
                                                   parent_link=parent_link,
                                                   parent_link_group=parent_link_group)
        self.wait_heartbeats()
        assert r.error.code == expected_response, \
            f'Got: \'{error_code_to_name(r.error.code)}\', ' \
            f'expected: \'{error_code_to_name(expected_response)}.\''
        if r.error.code == GiskardError.SUCCESS:
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
        god_map.collision_scene.reset_cache()
        collision_matrix = god_map.collision_scene.create_collision_matrix(collision_entries,
                                                                           defaultdict(lambda: 0.3))
        god_map.collision_scene.set_collision_matrix(collision_matrix)
        return god_map.collision_scene.check_collisions()

    def compute_all_collisions(self) -> Collisions:
        collision_entries = [CollisionEntry(type=CollisionEntry.AVOID_COLLISION,
                                            distance=-1)]
        return self.compute_collisions(collision_entries)

    def check_cpi_geq(self, links, distance_threshold, check_external=True, check_self=True):
        collisions = self.compute_all_collisions()
        links = [god_map.world.search_for_link_name(link_name) for link_name in links]
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
        links = [god_map.world.search_for_link_name(link_name) for link_name in links]
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
        try:
            p.header.frame_id = tf.get_tf_root()
        except AssertionError as e:
            p.header.frame_id = god_map.world.root_link_name
        p.pose.orientation.w = 1
        self.set_localization(p)
        self.wait_heartbeats()
        self.teleport_base(p)


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
