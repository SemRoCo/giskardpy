import json
from typing import Dict, Tuple, Optional, Union, List

import rospy
from actionlib import SimpleActionClient
from genpy import Message
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped, QuaternionStamped
from rospy import ServiceException
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import MarkerArray

from giskard_msgs.msg import MoveAction, MoveGoal, WorldBody, CollisionEntry, MoveResult, Constraint, \
    MoveCmd, MoveFeedback
from giskard_msgs.srv import DyeGroupRequest, DyeGroup, GetGroupInfoRequest, DyeGroupResponse
from giskard_msgs.srv import GetGroupNamesResponse, GetGroupInfoResponse, RegisterGroupRequest
from giskard_msgs.srv import RegisterGroupResponse
from giskard_msgs.srv import UpdateWorld, UpdateWorldRequest, UpdateWorldResponse, GetGroupInfo, \
    GetGroupNames, RegisterGroup
from giskardpy.exceptions import DuplicateNameException, UnknownGroupException
from giskardpy.goals.goal import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.model.utils import make_world_body_box
from giskardpy.my_types import goal_parameter
from giskardpy.utils.utils import position_dict_to_joint_states, convert_ros_message_to_dictionary, \
    replace_prefix_name_with_str


class GiskardWrapper:
    last_feedback: MoveFeedback = None

    def __init__(self, node_name: str = 'giskard'):
        giskard_topic = f'{node_name}/command'
        if giskard_topic is not None:
            self._client = SimpleActionClient(giskard_topic, MoveAction)
            self._update_world_srv = rospy.ServiceProxy(f'{node_name}/update_world', UpdateWorld)
            self._get_group_info_srv = rospy.ServiceProxy(f'{node_name}/get_group_info', GetGroupInfo)
            self._get_group_names_srv = rospy.ServiceProxy(f'{node_name}/get_group_names', GetGroupNames)
            self._register_groups_srv = rospy.ServiceProxy(f'{node_name}/register_groups', RegisterGroup)
            self._marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
            self.dye_group_srv = rospy.ServiceProxy(f'{node_name}/dye_group', DyeGroup)
            rospy.wait_for_service(f'{node_name}/update_world')
            self._client.wait_for_server()
        self.collisions = []
        self.clear_cmds()
        self._object_js_topics = {}
        rospy.sleep(.3)
        self.robot_name = self.get_group_names()[0]

    def register_group(self, new_group_name: str, root_link_name: str,
                       root_link_group_name: str) -> RegisterGroupResponse:
        """
        Register a new group for reference in collision checking. All child links of root_link_name will belong to it.
        :param new_group_name: Name of the new group.
        :param root_link_name: root link of the new group
        :param root_link_group_name: Name of the group root_link_name belongs to
        :return: RegisterGroupResponse
        """
        req = RegisterGroupRequest()
        req.group_name = new_group_name
        req.parent_group_name = root_link_group_name
        req.root_link_name = root_link_name
        res = self._register_groups_srv.call(req)  # type: RegisterGroupResponse
        if res.error_codes == res.DUPLICATE_GROUP_ERROR:
            raise DuplicateNameException(f'Group with name {new_group_name} already exists.')
        if res.error_codes == res.BUSY:
            raise ServiceException('Giskard is busy and can\'t process service call.')
        return res

    def _feedback_cb(self, msg: MoveFeedback):
        self.last_feedback = msg

    def set_cart_goal(self,
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
                      **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.set_json_goal(constraint_type='CartesianPose',
                           goal_pose=goal_pose,
                           tip_link=tip_link,
                           root_link=root_link,
                           root_group=root_group,
                           tip_group=tip_group,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           reference_linear_velocity=reference_linear_velocity,
                           reference_angular_velocity=reference_angular_velocity,
                           weight=weight,
                           **kwargs)

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
                               **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        In contrast to set_cart_goal, this tries to move the tip_link in a straight line to the goal_point.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.set_json_goal(constraint_type='CartesianPoseStraight',
                           goal_pose=goal_pose,
                           tip_link=tip_link,
                           tip_group=tip_group,
                           root_link=root_link,
                           root_group=root_group,
                           weight=weight,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           **kwargs)

    def set_translation_goal(self,
                             goal_point: PointStamped,
                             tip_link: str,
                             root_link: str,
                             tip_group: Optional[str] = None,
                             root_group: Optional[str] = None,
                             max_velocity: Optional[float] = None,
                             reference_velocity: Optional[float] = 0.2,
                             weight: float = WEIGHT_ABOVE_CA,
                             **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_point.
        :param goal_point:
        :param tip_link: tip link of the kinematic chain
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param max_velocity: m/s
        :param reference_velocity: m/s
        :param weight:
        """
        self.set_json_goal(constraint_type='CartesianPosition',
                           goal_point=goal_point,
                           tip_link=tip_link,
                           root_link=root_link,
                           tip_group=tip_group,
                           root_group=root_group,
                           max_velocity=max_velocity,
                           reference_velocity=reference_velocity,
                           weight=weight,
                           **kwargs)

    def set_straight_translation_goal(self,
                                      goal_pose: PoseStamped,
                                      tip_link: str,
                                      root_link: str,
                                      tip_group: Optional[str] = None,
                                      root_group: Optional[str] = None,
                                      reference_velocity: float = None,
                                      max_velocity: float = 0.2,
                                      weight: float = WEIGHT_ABOVE_CA,
                                      **kwargs: goal_parameter):
        """
        Same as set_translation_goal, but will try to move in a straight line.
        """
        self.set_json_goal(constraint_type='CartesianPositionStraight',
                           goal_pose=goal_pose,
                           tip_link=tip_link,
                           root_link=root_link,
                           tip_group=tip_group,
                           root_group=root_group,
                           reference_velocity=reference_velocity,
                           max_velocity=max_velocity,
                           weight=weight,
                           **kwargs)

    def set_rotation_goal(self,
                          goal_orientation: QuaternionStamped,
                          tip_link: str,
                          root_link: str,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          reference_velocity: Optional[float] = None,
                          max_velocity: Optional[float] = None,
                          weight=WEIGHT_ABOVE_CA,
                          **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_orientation.
        :param goal_orientation:
        :param tip_link: tip link of kinematic chain
        :param root_link: root link of kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: rad/s, approx limit
        :param max_velocity: rad/s, strict limit, but will slow the system down
        :param weight:
        """
        self.set_json_goal(constraint_type='CartesianOrientation',
                           goal_orientation=goal_orientation,
                           tip_link=tip_link,
                           root_link=root_link,
                           tip_group=tip_group,
                           root_group=root_group,
                           reference_velocity=reference_velocity,
                           max_velocity=max_velocity,
                           weight=weight,
                           **kwargs)

    def set_joint_goal(self,
                       goal_state: Dict[str, float],
                       group_name: Optional[str] = None,
                       weight: Optional[float] = None,
                       max_velocity: Optional[float] = None,
                       hard: bool = False,
                       **kwargs: goal_parameter):
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints
        :param hard: turns this into a hard constraint.
        """
        self.set_json_goal(constraint_type='JointPositionList',
                           goal_state=goal_state,
                           group_name=group_name,
                           weight=weight,
                           max_velocity=max_velocity,
                           hard=hard,
                           **kwargs)

    def set_align_planes_goal(self,
                              goal_normal: Vector3Stamped,
                              tip_link: str,
                              tip_normal: Vector3Stamped,
                              root_link: str,
                              tip_group: str = None,
                              root_group: str = None,
                              max_angular_velocity: Optional[float] = None,
                              weight: Optional[float] = None,
                              **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param goal_normal:
        :param tip_link: tip link of the kinematic chain
        :param tip_normal:
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip_link is not unique, search in this group for matches.
        :param root_group: if root_link is not unique, search in this group for matches.
        :param max_angular_velocity: rad/s
        :param weight:
        """
        self.set_json_goal(constraint_type='AlignPlanes',
                           tip_link=tip_link,
                           tip_group=tip_group,
                           tip_normal=tip_normal,
                           root_link=root_link,
                           root_group=root_group,
                           goal_normal=goal_normal,
                           max_angular_velocity=max_angular_velocity,
                           weight=weight,
                           **kwargs)

    def set_prediction_horizon(self, prediction_horizon: int, **kwargs: goal_parameter):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        self.set_json_goal(constraint_type='SetPredictionHorizon',
                           prediction_horizon=prediction_horizon,
                           **kwargs)

    def set_max_traj_length(self, new_length: float, **kwargs: goal_parameter):
        """
        Overwrites Giskard trajectory length limit for planning.
        If the trajectory is longer than new_length, Giskard will prempt the goal.
        :param new_length: in seconds
        """
        self.set_json_goal(constraint_type='SetMaxTrajLength',
                           new_length=new_length,
                           **kwargs)

    def set_limit_cartesian_velocity_goal(self,
                                          tip_link: str,
                                          root_link: str,
                                          tip_group: Optional[str] = None,
                                          root_group: Optional[str] = None,
                                          max_linear_velocity: float = 0.1,
                                          max_angular_velocity: float = 0.5,
                                          weight: float = WEIGHT_ABOVE_CA,
                                          hard: bool = False,
                                          **kwargs: goal_parameter):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param root_group: if the root_link is not unique, use this to say to which group the link belongs
        :param tip_group: if the tip_link is not unique, use this to say to which group the link belongs
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        self.set_json_goal('CartesianVelocityLimit',
                           root_link=root_link,
                           root_group=root_group,
                           tip_link=tip_link,
                           tip_group=tip_group,
                           weight=weight,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           hard=hard,
                           **kwargs)

    def set_grasp_bar_goal(self,
                           bar_center: PointStamped,
                           bar_axis: Vector3Stamped,
                           bar_length: float,
                           tip_link: str,
                           tip_grasp_axis: Vector3Stamped,
                           root_link: str,
                           tip_group: Optional[str] = None,
                           root_group: Optional[str] = None,
                           reference_linear_velocity: Optional[float] = None,
                           reference_angular_velocity: Optional[float] = None,
                           weight: float = WEIGHT_ABOVE_CA,
                           **kwargs: goal_parameter):
        """
        Like a CartesianPose but with more freedom.
        tip_link is allowed to be at any point along bar_axis, that is without bar_center +/- bar_length.
        It will align tip_grasp_axis with bar_axis, but allows rotation around it.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param tip_grasp_axis: axis of tip_link that will be aligned with bar_axis
        :param bar_center: center of the bar to be grasped
        :param bar_axis: alignment of the bar to be grasped
        :param bar_length: length of the bar to be grasped
        :param root_group: if root_link is not unique, search in this group for matches
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight:
        """
        self.set_json_goal(constraint_type='GraspBar',
                           root_link=root_link,
                           tip_link=tip_link,
                           tip_grasp_axis=tip_grasp_axis,
                           bar_center=bar_center,
                           bar_axis=bar_axis,
                           bar_length=bar_length,
                           root_group=root_group,
                           tip_group=tip_group,
                           reference_linear_velocity=reference_linear_velocity,
                           reference_angular_velocity=reference_angular_velocity,
                           weight=weight,
                           **kwargs)

    def set_open_container_goal(self,
                                tip_link: str,
                                environment_link: str,
                                tip_group: Optional[str] = None,
                                environment_group: Optional[str] = None,
                                goal_joint_state: Optional[float] = None,
                                weight=WEIGHT_ABOVE_CA, ):
        """
        Open a container in an environment.
        Only works with the environment was added as urdf.
        Assumes that a handle has already been grasped.
        Can only handle containers with 1 dof, e.g. drawers or doors.
        :param tip_link: end effector that is grasping the handle
        :param environment_link: name of the handle that was grasped
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param environment_group: if environment_link is not unique, search in this group for matches
        :param goal_joint_state: goal state for the container. default is maximum joint state.
        :param weight:
        """
        self.set_json_goal('Open',
                           tip_link=tip_link,
                           environment_link=environment_link,
                           tip_group=tip_group,
                           environment_group=environment_group,
                           goal_joint_state=goal_joint_state,
                           weight=weight)

    def set_close_container_goal(self,
                                 tip_link: str,
                                 environment_link: str,
                                 tip_group: Optional[str] = None,
                                 environment_group: Optional[str] = None,
                                 goal_joint_state: Optional[float] = None,
                                 weight=WEIGHT_ABOVE_CA, ):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.set_json_goal('Close',
                           tip_link=tip_link,
                           environment_link=environment_link,
                           tip_group=tip_group,
                           environment_group=environment_group,
                           goal_joint_state=goal_joint_state,
                           weight=weight)

    def set_pointing_goal(self,
                          goal_point: PointStamped,
                          tip_link: str,
                          pointing_axis: Vector3Stamped,
                          root_link: str,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          max_velocity: float = 0.3,
                          weight: float = WEIGHT_BELOW_CA,
                          **kwargs: goal_parameter):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param tip_group: if tip_link is not unique, search this group for matches.
        :param root_group: if root_link is not unique, search this group for matches.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        """
        self.set_json_goal(constraint_type='Pointing',
                           tip_link=tip_link,
                           tip_group=tip_group,
                           goal_point=goal_point,
                           root_link=root_link,
                           root_group=root_group,
                           pointing_axis=pointing_axis,
                           max_velocity=max_velocity,
                           weight=weight,
                           **kwargs)

    def set_json_goal(self,
                      constraint_type: str,
                      **kwargs: goal_parameter):
        """
        Set a goal for any of the goals defined in any of the files in src/giskardpy/goals/
        :param constraint_type: Name of the Goal class
        :param kwargs: maps constraint parameter names to their values.
                        Values should be a float, str, dict or ros messages.
        """
        constraint = Constraint()
        constraint.type = constraint_type
        for k, v in kwargs.copy().items():
            if v is None:
                del kwargs[k]
            else:
                kwargs[k] = self.parse_messages(v)
        kwargs = replace_prefix_name_with_str(kwargs)
        constraint.parameter_value_pair = json.dumps(kwargs)
        self.cmd_seq[-1].constraints.append(constraint)

    def parse_messages(self,
                       val):

        if isinstance(val, List):
            for i, element in enumerate(val):
                val[i] = self.parse_messages(element)

        elif isinstance(val, Dict):
            for k, v in val.copy().items():
                val[k] = self.parse_messages(v)

        elif isinstance(val, Message):
            return convert_ros_message_to_dictionary(val)

        return val

    def _set_collision_entries(self, collisions: List[CollisionEntry]):
        """
        Adds collision entries to the current goal
        :param collisions: list of CollisionEntry
        """
        self.cmd_seq[-1].collisions.extend(collisions)

    def allow_collision(self, group1: str = CollisionEntry.ALL, group2: str = CollisionEntry.ALL):
        """
        Tell Giskard to allow collision between group1 and group2. Use CollisionEntry.ALL to allow collision with all
        groups.
        :param group1: name of the first group
        :param group2: name of the second group
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = str(group1)
        collision_entry.group2 = str(group2)
        self._set_collision_entries([collision_entry])

    def avoid_collision(self,
                        min_distance: Optional[float] = None,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL):
        """
        Tell Giskard to avoid collision between group1 and group2. Use CollisionEntry.ALL to allow collision with all
        groups.
        :param min_distance: set this to overwrite the default distances
        :param group1: name of the first group
        :param group2: name of the second group
        """
        if min_distance is None:
            min_distance = - 1
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        collision_entry.group1 = group1
        collision_entry.group2 = group2
        self._set_collision_entries([collision_entry])

    def allow_all_collisions(self):
        """
        Allows all collisions for next goal.
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        self._set_collision_entries([collision_entry])

    def set_avoid_joint_limits_goal(self,
                                    percentage: int = 15,
                                    joint_list: Optional[List[str]] = None,
                                    weight: Optional[float] = None):
        """
        This goal will push joints away from their position limits. For example if percentage is 15 and the joint
        limits are 0-100, it will push it into the 15-85 range.
        """
        self.set_json_goal(constraint_type='AvoidJointLimits',
                           percentage=percentage,
                           weight=weight,
                           joint_list=joint_list)

    def allow_self_collision(self, robot_name: Optional[str] = None):
        """
        Allows the collision of the robot with itself for the next goal.
        :param robot_name: if there are multiple robots, specify which one.
        """
        if robot_name is None:
            robot_name = self.robot_name
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = robot_name
        collision_entry.group2 = robot_name
        self._set_collision_entries([collision_entry])

    def avoid_self_collision(self, robot_name: Optional[str] = None, min_distance: Optional[float] = None):
        """
        Avoid collisions with itself for the next goal.
        :param robot_name: if there are multiple robots, specify one
        :param min_distance: set to overwrite default distances
        """
        if min_distance is None:
            min_distance = -1
        if robot_name is None:
            robot_name = self.robot_name
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.group1 = robot_name
        collision_entry.group2 = robot_name
        collision_entry.distance = min_distance
        self._set_collision_entries([collision_entry])

    def avoid_all_collisions(self, min_distance: Optional[float] = None):
        """
        Avoids all collisions for next goal.
        If you don't want to override the distance, don't call this function. Avoid all is the default, if you don't
        add any collision entries.
        :param min_distance: set this to overwrite default distances
        """
        if min_distance is None:
            min_distance = -1
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        self._set_collision_entries([collision_entry])

    def add_cmd(self):
        """
        Adds another command to the goal sequence. Any set_something_goal calls will be added this new goal.
        This is used, if you want Giskard to plan multiple goals in succession.
        """
        move_cmd = MoveCmd()
        self.cmd_seq.append(move_cmd)

    def clear_cmds(self):
        """
        Removes all move commands from the current goal, collision entries are left untouched.
        """
        self.cmd_seq = []
        self.add_cmd()

    @property
    def number_of_cmds(self):
        return len(self.cmd_seq)

    def plan_and_execute(self, wait: bool = True) -> MoveResult:
        """
        :param wait: this function blocks if wait=True
        :return: result from giskard
        """
        return self.send_goal(MoveGoal.PLAN_AND_EXECUTE, wait)

    def plan(self, wait: bool = True) -> MoveResult:
        """
        Plans, but doesn't execute the goal. Useful, if you just want to look at the planning ghost.
        :param wait: this function blocks if wait=True
        :return: result from Giskard
        """
        return self.send_goal(MoveGoal.PLAN_ONLY, wait)

    def send_goal(self, goal_type: int, wait: bool = True) -> Optional[MoveResult]:
        """
        Send goal to Giskard. Use this if you want to specify the goal_type, otherwise stick to wrappers like
        plan_and_execute.
        :param goal_type: one of the constants in MoveGoal
        :param wait: blocks if wait=True
        :return: result from Giskard
        """
        goal = self._get_goal()
        goal.type = goal_type
        if wait:
            self._client.send_goal_and_wait(goal)
            return self._client.get_result()
        else:
            self._client.send_goal(goal, feedback_cb=self._feedback_cb)

    def get_collision_entries(self) -> List[CollisionEntry]:
        return self.cmd_seq

    def _get_goal(self) -> MoveGoal:
        goal = MoveGoal()
        goal.cmd_seq = self.cmd_seq
        goal.type = MoveGoal.PLAN_AND_EXECUTE
        self.clear_cmds()
        return goal

    def interrupt(self):
        """
        Stops the goal that was last sent to Giskard.
        """
        self._client.cancel_goal()

    def cancel_all_goals(self):
        """
        Stops any goal that Giskard is processing and attempts to halt the robot, even those not send from this client.
        """
        self._client.cancel_all_goals()

    def get_result(self, timeout: rospy.Duration = rospy.Duration()) -> MoveResult:
        """
        Waits for Giskard result and returns it. Only used when plan_and_execute was called with wait=False
        :param timeout: how long to wait
        """
        if not self._client.wait_for_result(timeout):
            raise TimeoutError('Timeout while waiting for goal.')
        return self._client.get_result()

    def clear_world(self, timeout: float = 2) -> UpdateWorldResponse:
        """
        Resets the world to what it was when Giskard was launched.
        """
        req = UpdateWorldRequest()
        req.operation = UpdateWorldRequest.REMOVE_ALL
        req.timeout = timeout
        result: UpdateWorldResponse = self._update_world_srv.call(req)
        if result.error_codes == UpdateWorldResponse.SUCCESS:
            self._object_js_topics = {}
        return result

    def remove_group(self,
                     name: str,
                     timeout: float = 2) -> UpdateWorldResponse:
        """
        Removes a group and all links and joints it contains from the world.
        Be careful, you can remove parts of the robot like that.
        """
        world_body = WorldBody()
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.REMOVE
        req.timeout = timeout
        req.body = world_body
        result: UpdateWorldResponse = self._update_world_srv.call(req)
        if result.error_codes == UpdateWorldResponse.SUCCESS:
            if name in self._object_js_topics:
                del self._object_js_topics[name]
        return result

    def add_box(self,
                name: str,
                size: Tuple[float, float, float],
                pose: PoseStamped,
                parent_link: str = '',
                parent_link_group: str = '',
                timeout: float = 2) -> UpdateWorldResponse:
        """
        Adds a new box to the world tree and attaches it to parent_link.
        If parent_link_group and parent_link are empty, the box will be attached to the world root link, e.g., map.
        :param name: How the new group will be called
        :param size: X, Y and Z dimensions of the box, respectively
        :param pose: Where the root link of the new object will be positioned
        :param parent_link: Name of the link, the object will get attached to
        :param parent_link_group: Name of the group in which Giskard will search for parent_link
        :param timeout: Can wait this many seconds, in case Giskard is busy
        :return: Response message of the service call
        """
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.ADD
        req.timeout = timeout
        req.body = make_world_body_box(size[0], size[1], size[2])
        req.parent_link_group = parent_link_group
        req.parent_link = parent_link
        req.pose = pose
        return self._update_world_srv.call(req)

    def add_sphere(self,
                   name: str,
                   radius: float,
                   pose: PoseStamped,
                   parent_link: str = '',
                   parent_link_group: str = '',
                   timeout: float = 2) -> UpdateWorldResponse:
        """
        See add_box.
        """
        world_body = WorldBody()
        world_body.type = WorldBody.PRIMITIVE_BODY
        world_body.shape.type = SolidPrimitive.SPHERE
        world_body.shape.dimensions.append(radius)
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.ADD
        req.timeout = timeout
        req.body = world_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._update_world_srv.call(req)

    def add_mesh(self,
                 name: str,
                 mesh: str,
                 pose: PoseStamped,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 scale: Tuple[float, float, float] = (1, 1, 1),
                 timeout: float = 2) -> UpdateWorldResponse:
        """
        See add_box.
        :param mesh: path to the mesh location, can be ros package path, e.g.,
                        package://giskardpy/test/urdfs/meshes/bowl_21.obj
        """
        world_body = WorldBody()
        world_body.type = WorldBody.MESH_BODY
        world_body.mesh = mesh
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.ADD
        req.timeout = timeout
        req.body = world_body
        req.pose = pose
        req.body.scale.x = scale[0]
        req.body.scale.y = scale[1]
        req.body.scale.z = scale[2]
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._update_world_srv.call(req)

    def add_cylinder(self,
                     name: str,
                     height: float,
                     radius: float,
                     pose: PoseStamped,
                     parent_link: str = '',
                     parent_link_group: str = '',
                     timeout: float = 2) -> UpdateWorldResponse:
        """
        See add_box.
        """
        world_body = WorldBody()
        world_body.type = WorldBody.PRIMITIVE_BODY
        world_body.shape.type = SolidPrimitive.CYLINDER
        world_body.shape.dimensions = [0, 0]
        world_body.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT] = height
        world_body.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS] = radius
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.ADD
        req.timeout = timeout
        req.body = world_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._update_world_srv.call(req)

    def update_parent_link_of_group(self,
                                    name: str,
                                    parent_link: str,
                                    parent_link_group: Optional[str] = '',
                                    timeout: float = 2) -> UpdateWorldResponse:
        """
        Removes the joint connecting the root link of a group and attaches it to a parent_link.
        The object will not move relative to the world's root link in this process.
        :param name: name of the group
        :param parent_link: name of the new parent link
        :param parent_link_group: if parent_link is not unique, search in this group for matches.
        :param timeout: how long to wait in case Giskard is busy processing a goal.
        :return: result message
        """
        req = UpdateWorldRequest()
        req.operation = UpdateWorldRequest.UPDATE_PARENT_LINK
        req.group_name = str(name)
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        req.timeout = timeout
        return self._update_world_srv.call(req)

    def detach_group(self, object_name: str, timeout: float = 2):
        """
        A wrapper for update_parent_link_of_group which set parent_link to the root link of the world.
        """
        req = UpdateWorldRequest()
        req.timeout = timeout
        req.group_name = str(object_name)
        req.operation = req.UPDATE_PARENT_LINK
        return self._update_world_srv.call(req)

    def add_urdf(self,
                 name: str,
                 urdf: str,
                 pose: PoseStamped,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 js_topic: Optional[str] = '',
                 set_js_topic: Optional[str] = '',
                 timeout: float = 2) -> UpdateWorldResponse:
        """
        Adds a urdf to the world.
        :param name: name the group containing the urdf will have.
        :param urdf: urdf as string, no path!
        :param pose: pose of the root link of the new object
        :param parent_link: to which link the urdf will be attached
        :param parent_link_group: if parent_link is not unique, search here for matches.
        :param js_topic: Giskard will listen on that topic for joint states and update the urdf accordingly
        :param set_js_topic: A topic that the python wrapper will use to set the urdf joint state.
                                Only works if there is, e.g., a joint_state_publisher that listens to it.
        :param timeout: how long to wait if Giskard is busy.
        :return: response message
        """
        js_topic = str(js_topic)
        if set_js_topic == '':
            set_js_topic = js_topic
        urdf_body = WorldBody()
        urdf_body.type = WorldBody.URDF_BODY
        urdf_body.urdf = str(urdf)
        urdf_body.joint_state_topic = js_topic
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.ADD
        req.timeout = timeout
        req.body = urdf_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        if set_js_topic:
            # FIXME publisher has to be removed, when object gets deleted
            # FIXME there could be sync error, if objects get added/removed by something else
            self._object_js_topics[name] = rospy.Publisher(set_js_topic, JointState, queue_size=10)
        return self._update_world_srv.call(req)

    def set_object_joint_state(self, object_name: str, joint_states: Union[JointState, dict]):
        """
        If add_urdf was used with a set_js_topic, you can use this to send out a message.
        :param object_name: name of the group
        :param joint_states: joint state message or a dict that maps joint name to position
        """
        if isinstance(joint_states, dict):
            joint_states = position_dict_to_joint_states(joint_states)
        self._object_js_topics[object_name].publish(joint_states)

    def dye_group(self, group_name: str, rgba: Tuple[float, float, float, float]) -> DyeGroupResponse:
        """
        Change the color of the ghost for this particular group.
        """
        req = DyeGroupRequest()
        req.group_name = group_name
        req.color.r = rgba[0]
        req.color.g = rgba[1]
        req.color.b = rgba[2]
        req.color.a = rgba[3]
        return self.dye_group_srv(req)

    def get_group_names(self) -> List[str]:
        """
        Returns the names of every group in the world.
        """
        resp = self._get_group_names_srv()  # type: GetGroupNamesResponse
        return resp.group_names

    def get_group_info(self, group_name: str) -> GetGroupInfoResponse:
        """
        Returns the joint state, joint state topic and pose of a group.
        """
        req = GetGroupInfoRequest()
        req.group_name = group_name
        return self._get_group_info_srv.call(req)

    def get_controlled_joints(self, group_name: str) -> List[str]:
        """
        Returns all joints of a group that are flagged as controlled.
        """
        return self.get_group_info(group_name).controlled_joints

    def update_group_pose(self, group_name: str, new_pose: PoseStamped, timeout: float = 2) -> UpdateWorldResponse:
        """
        Overwrites the pose specified in the joint that connects the two groups.
        :param group_name: Name of the group that will move
        :param new_pose: New pose of the group
        :param timeout: How long to wait if Giskard is busy
        :return: Giskard's reply
        """
        req = UpdateWorldRequest()
        req.operation = req.UPDATE_POSE
        req.group_name = group_name
        req.pose = new_pose
        req.timeout = timeout
        res = self._update_world_srv.call(req)
        if res.error_codes == UpdateWorldResponse.SUCCESS:
            return res
        if res.error_codes == UpdateWorldResponse.UNKNOWN_GROUP_ERROR:
            raise UnknownGroupException(res.error_msg)
        raise ServiceException(res.error_msg)
