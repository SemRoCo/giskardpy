from collections import defaultdict
from typing import Dict, Tuple, Optional, List
import numpy as np
import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped, QuaternionStamped
from rospy import ServiceException
from shape_msgs.msg import SolidPrimitive

import giskard_msgs.msg as giskard_msgs
from giskard_msgs.msg import MoveAction, MoveGoal, WorldBody, CollisionEntry, MoveResult, MoveFeedback, MotionGoal, \
    Monitor, WorldGoal, WorldAction, WorldResult, GiskardError
from giskard_msgs.srv import DyeGroupRequest, DyeGroup, GetGroupInfoRequest, DyeGroupResponse
from giskard_msgs.srv import GetGroupInfo, GetGroupNames
from giskard_msgs.srv import GetGroupNamesResponse, GetGroupInfoResponse
from giskardpy.data_types import goal_parameter
from giskardpy.exceptions import DuplicateNameException, UnknownGroupException
from giskardpy.goals.align_planes import AlignPlanes
from giskardpy.goals.align_to_push_door import AlignToPushDoor
from giskardpy.goals.base_traj_follower import CarryMyBullshit
from giskardpy.goals.cartesian_goals import CartesianPose, DiffDriveBaseGoal, CartesianVelocityLimit, \
    CartesianOrientation, CartesianPoseStraight, CartesianPosition, CartesianPositionStraight
from giskardpy.goals.collision_avoidance import CollisionAvoidance
from giskardpy.goals.grasp_bar import GraspBar
from giskardpy.goals.joint_goals import JointPositionList, AvoidJointLimits, SetSeedConfiguration, SetOdometry
from giskardpy.goals.open_close import Close, Open
from giskardpy.goals.pointing import Pointing
from giskardpy.goals.pre_push_door import PrePushDoor
from giskardpy.goals.realtime_goals import RealTimePointing
from giskardpy.goals.set_prediction_horizon import SetPredictionHorizon
from giskardpy.model.utils import make_world_body_box
from giskardpy.monitors.cartesian_monitors import PoseReached, PositionReached, OrientationReached, PointingAt, \
    VectorsAligned, DistanceToLine
from giskardpy.monitors.joint_monitors import JointGoalReached
from giskardpy.monitors.monitors import LocalMinimumReached, TimeAbove, Alternator, CancelMotion, EndMotion
from giskardpy.monitors.payload_monitors import Print, Sleep, SetMaxTrajectoryLength, \
    UpdateParentLinkOfGroup, PayloadAlternator
from giskardpy.utils.utils import kwargs_to_json, get_all_classes_in_package


class WorldWrapper:
    def __init__(self, node_name: str):
        self._get_group_info_srv = rospy.ServiceProxy(f'{node_name}/get_group_info', GetGroupInfo)
        self._get_group_names_srv = rospy.ServiceProxy(f'{node_name}/get_group_names', GetGroupNames)
        self._dye_group_srv = rospy.ServiceProxy(f'{node_name}/dye_group', DyeGroup)
        self._client = SimpleActionClient(f'{node_name}/update_world', WorldAction)
        self._client.wait_for_server()
        rospy.wait_for_service(self._get_group_names_srv.resolved_name)
        self.robot_name = self.get_group_names()[0]

    def clear(self) -> WorldResult:
        """
        Resets the world to what it was when Giskard was launched.
        """
        req = WorldGoal()
        req.operation = WorldGoal.REMOVE_ALL
        return self._send_goal_and_wait(req)

    def remove_group(self, name: str) -> WorldResult:
        """
        Removes a group and all links and joints it contains from the world.
        Be careful, you can remove parts of the robot like that.
        """
        world_body = WorldBody()
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.REMOVE
        req.body = world_body
        return self._send_goal_and_wait(req)

    def _send_goal_and_wait(self, goal: WorldGoal) -> WorldResult:
        self._client.send_goal_and_wait(goal)
        return self._client.get_result()

    def add_box(self,
                name: str,
                size: Tuple[float, float, float],
                pose: PoseStamped,
                parent_link: str = '',
                parent_link_group: str = '') -> WorldResult:
        """
        Adds a new box to the world tree and attaches it to parent_link.
        If parent_link_group and parent_link are empty, the box will be attached to the world root link, e.g., map.
        :param name: How the new group will be called
        :param size: X, Y and Z dimensions of the box, respectively
        :param pose: Where the root link of the new object will be positioned
        :param parent_link: Name of the link, the object will get attached to
        :param parent_link_group: Name of the group in which Giskard will search for parent_link
        :return: Response message of the service call
        """
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = make_world_body_box(size[0], size[1], size[2])
        req.parent_link_group = parent_link_group
        req.parent_link = parent_link
        req.pose = pose
        return self._send_goal_and_wait(req)

    def add_sphere(self,
                   name: str,
                   radius: float,
                   pose: PoseStamped,
                   parent_link: str = '',
                   parent_link_group: str = '') -> WorldResult:
        """
        See add_box.
        """
        world_body = WorldBody()
        world_body.type = WorldBody.PRIMITIVE_BODY
        world_body.shape.type = SolidPrimitive.SPHERE
        world_body.shape.dimensions.append(radius)
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = world_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

    def add_mesh(self,
                 name: str,
                 mesh: str,
                 pose: PoseStamped,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 scale: Tuple[float, float, float] = (1, 1, 1)) -> WorldResult:
        """
        See add_box.
        :param mesh: path to the mesh location, can be ros package path, e.g.,
                        package://giskardpy/test/urdfs/meshes/bowl_21.obj
        """
        world_body = WorldBody()
        world_body.type = WorldBody.MESH_BODY
        world_body.mesh = mesh
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = world_body
        req.pose = pose
        req.body.scale.x = scale[0]
        req.body.scale.y = scale[1]
        req.body.scale.z = scale[2]
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

    def add_cylinder(self,
                     name: str,
                     height: float,
                     radius: float,
                     pose: PoseStamped,
                     parent_link: str = '',
                     parent_link_group: str = '') -> WorldResult:
        """
        See add_box.
        """
        world_body = WorldBody()
        world_body.type = WorldBody.PRIMITIVE_BODY
        world_body.shape.type = SolidPrimitive.CYLINDER
        world_body.shape.dimensions = [0, 0]
        world_body.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT] = height
        world_body.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS] = radius
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = world_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

    def update_parent_link_of_group(self,
                                    name: str,
                                    parent_link: str,
                                    parent_link_group: Optional[str] = '') -> WorldResult:
        """
        Removes the joint connecting the root link of a group and attaches it to a parent_link.
        The object will not move relative to the world's root link in this process.
        :param name: name of the group
        :param parent_link: name of the new parent link
        :param parent_link_group: if parent_link is not unique, search in this group for matches.
        :param timeout: how long to wait in case Giskard is busy processing a goal.
        :return: result message
        """
        req = WorldGoal()
        req.operation = WorldGoal.UPDATE_PARENT_LINK
        req.group_name = str(name)
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

    def detach_group(self, object_name: str) -> WorldResult:
        """
        A wrapper for update_parent_link_of_group which set parent_link to the root link of the world.
        """
        req = WorldGoal()
        req.group_name = str(object_name)
        req.operation = req.UPDATE_PARENT_LINK
        return self._send_goal_and_wait(req)

    def add_urdf(self,
                 name: str,
                 urdf: str,
                 pose: PoseStamped,
                 parent_link: str = '',
                 parent_link_group: str = '',
                 js_topic: Optional[str] = '') -> WorldResult:
        """
        Adds a urdf to the world.
        :param name: name the group containing the urdf will have.
        :param urdf: urdf as string, no path!
        :param pose: pose of the root link of the new object
        :param parent_link: to which link the urdf will be attached
        :param parent_link_group: if parent_link is not unique, search here for matches.
        :param js_topic: Giskard will listen on that topic for joint states and update the urdf accordingly
        :return: response message
        """
        js_topic = str(js_topic)
        urdf_body = WorldBody()
        urdf_body.type = WorldBody.URDF_BODY
        urdf_body.urdf = str(urdf)
        urdf_body.joint_state_topic = js_topic
        req = WorldGoal()
        req.group_name = str(name)
        req.operation = WorldGoal.ADD
        req.body = urdf_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._send_goal_and_wait(req)

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
        return self._dye_group_srv(req)

    def get_group_names(self) -> List[str]:
        """
        Returns the names of every group in the world.
        """
        resp: GetGroupNamesResponse = self._get_group_names_srv()
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

    def update_group_pose(self, group_name: str, new_pose: PoseStamped) -> WorldResult:
        """
        Overwrites the pose specified in the joint that connects the two groups.
        :param group_name: Name of the group that will move
        :param new_pose: New pose of the group
        :return: Giskard's reply
        """
        req = WorldGoal()
        req.operation = req.UPDATE_POSE
        req.group_name = group_name
        req.pose = new_pose
        res = self._send_goal_and_wait(req)
        if res.error.code == GiskardError.SUCCESS:
            return res
        if res.error.code == GiskardError.UNKNOWN_GROUP:
            raise UnknownGroupException(res.error.msg)
        raise ServiceException(res.error.msg)

    def register_group(self, new_group_name: str, root_link_name: str,
                       root_link_group_name: str) -> WorldResult:
        """
        Register a new group for reference in collision checking. All child links of root_link_name will belong to it.
        :param new_group_name: Name of the new group.
        :param root_link_name: root link of the new group
        :param root_link_group_name: Name of the group root_link_name belongs to
        :return: WorldResult
        """
        req = WorldGoal()
        req.operation = WorldGoal.REGISTER_GROUP
        req.group_name = new_group_name
        req.parent_link_group = root_link_group_name
        req.parent_link = root_link_name
        res = self._send_goal_and_wait(req)
        if res.error.code == GiskardError.DUPLICATE_NAME:
            raise DuplicateNameException(f'Group with name {new_group_name} already exists.')
        return res


class MotionGoalWrapper:
    _goals: List[MotionGoal]
    _collision_entries: Dict[Tuple[str, str, str], List[CollisionEntry]]
    avoid_name_conflict: bool

    def __init__(self, robot_name: str, avoid_name_conflict: bool = False):
        self.robot_name = robot_name
        self.reset()
        self.avoid_name_conflict = avoid_name_conflict

    def reset(self):
        """
        Clears all goals.
        """
        self._goals = []
        self._collision_entries = defaultdict(list)

    def get_goals(self) -> List[MotionGoal]:
        self._add_collision_entries_as_goals()
        return self._goals

    def number_of_goals(self) -> int:
        return len(self._goals)

    def add_motion_goal(self, *,
                        motion_goal_class: str,
                        name: Optional[str] = None,
                        start_condition: str = '',
                        hold_condition: str = '',
                        end_condition: str = '',
                        **kwargs):
        """
        Generic function to add a motion goal.
        :param motion_goal_class: Name of a class defined in src/giskardpy/goals
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param hold_condition: a logical expression. Goal will be on hold if it is True and active otherwise
        :param end_condition: a logical expression. Goal will become inactive when this becomes True.
        :param kwargs: kwargs for __init__ function of motion_goal_class
        """
        name = name or motion_goal_class
        if self.avoid_name_conflict:
            name = f'G{self.number_of_goals()} {name}'
        motion_goal = MotionGoal()
        motion_goal.name = name
        motion_goal.motion_goal_class = motion_goal_class
        motion_goal.start_condition = start_condition
        motion_goal.hold_condition = hold_condition
        motion_goal.end_condition = end_condition
        motion_goal.kwargs = kwargs_to_json(kwargs)
        self._goals.append(motion_goal)

    def _add_collision_avoidance(self,
                                 collisions: List[CollisionEntry],
                                 start_condition: str = '',
                                 hold_condition: str = '',
                                 end_condition: str = ''):
        key = (start_condition, hold_condition, end_condition)
        self._collision_entries[key].extend(collisions)

    def _add_collision_entries_as_goals(self):
        for (start_condition, hold_condition, end_condition), collision_entries in self._collision_entries.items():
            name = 'collision avoidance'
            if start_condition or hold_condition or end_condition:
                name += f'{start_condition}, {hold_condition}, {end_condition}'
            self.add_motion_goal(motion_goal_class=CollisionAvoidance.__name__,
                                 name=name,
                                 collision_entries=collision_entries,
                                 start_condition=start_condition,
                                 hold_condition=hold_condition,
                                 end_condition=end_condition)

    def allow_collision(self,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL,
                        start_condition: str = '',
                        hold_condition: str = '',
                        end_condition: str = ''):
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
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def avoid_collision(self,
                        min_distance: Optional[float] = None,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL,
                        start_condition: str = '',
                        hold_condition: str = '',
                        end_condition: str = ''):
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
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def allow_all_collisions(self,
                             start_condition: str = '',
                             hold_condition: str = '',
                             end_condition: str = ''):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def avoid_all_collisions(self,
                             min_distance: Optional[float] = None,
                             start_condition: str = '',
                             hold_condition: str = '',
                             end_condition: str = ''):
        """
        If you don't want to override the distance, don't call this function. Avoid all is the default, if you don't
        add any collision entries.
        :param min_distance: set this to overwrite default distances
        """
        if min_distance is None:
            min_distance = -1
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def allow_self_collision(self,
                             robot_name: Optional[str] = None,
                             start_condition: str = '',
                             hold_condition: str = '',
                             end_condition: str = ''):
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
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_condition=start_condition,
                                      hold_condition=hold_condition,
                                      end_condition=end_condition)

    def add_joint_position(self,
                           goal_state: Dict[str, float],
                           group_name: Optional[str] = None,
                           weight: Optional[float] = None,
                           max_velocity: Optional[float] = None,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           hold_condition: str = '',
                           end_condition: str = '',
                           **kwargs: goal_parameter):
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight: None = use default weight
        :param max_velocity: will be applied to all joints
        """
        self.add_motion_goal(motion_goal_class=JointPositionList.__name__,
                             goal_state=goal_state,
                             group_name=group_name,
                             weight=weight,
                             max_velocity=max_velocity,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_cartesian_pose(self,
                           goal_pose: PoseStamped,
                           tip_link: str,
                           root_link: str,
                           tip_group: Optional[str] = None,
                           root_group: Optional[str] = None,
                           reference_linear_velocity: Optional[float] = None,
                           reference_angular_velocity: Optional[float] = None,
                           absolute: bool = False,
                           weight: Optional[float] = None,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           hold_condition: str = '',
                           end_condition: str = '',
                           **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link to the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: None = use default weight
        """
        self.add_motion_goal(motion_goal_class=CartesianPose.__name__,
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             root_link=root_link,
                             root_group=root_group,
                             tip_group=tip_group,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             weight=weight,
                             name=name,
                             absolute=absolute,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_align_planes(self,
                         goal_normal: Vector3Stamped,
                         tip_link: str,
                         tip_normal: Vector3Stamped,
                         root_link: str,
                         tip_group: str = None,
                         root_group: str = None,
                         reference_angular_velocity: Optional[float] = None,
                         weight: Optional[float] = None,
                         name: Optional[str] = None,
                         start_condition: str = '',
                         hold_condition: str = '',
                         end_condition: str = '',
                         **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param goal_normal:
        :param tip_link: tip link of the kinematic chain
        :param tip_normal:
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip_link is not unique, search in this group for matches.
        :param root_group: if root_link is not unique, search in this group for matches.
        :param reference_angular_velocity: rad/s
        :param weight:
        """
        self.add_motion_goal(motion_goal_class=AlignPlanes.__name__,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             tip_normal=tip_normal,
                             root_link=root_link,
                             root_group=root_group,
                             goal_normal=goal_normal,
                             max_angular_velocity=reference_angular_velocity,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_avoid_joint_limits(self,
                               percentage: int = 15,
                               joint_list: Optional[List[str]] = None,
                               weight: Optional[float] = None,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               hold_condition: str = '',
                               end_condition: str = ''):
        """
        This goal will push joints away from their position limits. For example if percentage is 15 and the joint
        limits are 0-100, it will push it into the 15-85 range.
        """
        self.add_motion_goal(motion_goal_class=AvoidJointLimits.__name__,
                             percentage=percentage,
                             weight=weight,
                             joint_list=joint_list,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_close_container(self,
                            tip_link: str,
                            environment_link: str,
                            tip_group: Optional[str] = None,
                            environment_group: Optional[str] = None,
                            goal_joint_state: Optional[float] = None,
                            weight: Optional[float] = None,
                            name: Optional[str] = None,
                            start_condition: str = '',
                            hold_condition: str = '',
                            end_condition: str = ''):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.add_motion_goal(motion_goal_class=Close.__name__,
                             tip_link=tip_link,
                             environment_link=environment_link,
                             tip_group=tip_group,
                             environment_group=environment_group,
                             goal_joint_state=goal_joint_state,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_open_container(self,
                           tip_link: str,
                           environment_link: str,
                           tip_group: Optional[str] = None,
                           environment_group: Optional[str] = None,
                           goal_joint_state: Optional[float] = None,
                           weight: Optional[float] = None,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           hold_condition: str = '',
                           end_condition: str = ''):
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
        self.add_motion_goal(motion_goal_class=Open.__name__,
                             tip_link=tip_link,
                             environment_link=environment_link,
                             tip_group=tip_group,
                             environment_group=environment_group,
                             goal_joint_state=goal_joint_state,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_align_to_push_door(self,
                               root_link: str,
                               tip_link: str,
                               door_object: str,
                               door_handle: str,
                               tip_gripper_axis: Vector3Stamped,
                               weight: float,
                               tip_group: Optional[str] = None,
                               root_group: Optional[str] = None,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               hold_condition: str = '',
                               end_condition: str = ''):
        """
        Aligns the tip_link with the door_object to push it open. Only works if the door object is part of the urdf.
        The door has to be open a little before aligning.
        : param root_link: root link of the kinematic chain
        : param tip_link: end effector
        : param door object: name of the object to be pushed
        : param door_height: height of the door
        : param door_handle: name of the object handle
        : param object_joint_name: name of the joint that rotates
        : param tip_gripper_axis: axis of the tip_link that will be aligned along the door rotation axis
        : param object_rotation_axis: door rotation axis w.r.t root
        """
        self.add_motion_goal(motion_goal_class=AlignToPushDoor.__name__,
                             root_link=root_link,
                             tip_link=tip_link,
                             door_handle=door_handle,
                             door_object=door_object,
                             tip_gripper_axis=tip_gripper_axis,
                             tip_group=tip_group,
                             root_group=root_group,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_pre_push_door(self,
                          root_link: str,
                          tip_link: str,
                          door_object: str,
                          door_handle: str,
                          weight: float,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          reference_linear_velocity: Optional[float] = None,
                          reference_angular_velocity: Optional[float] = None,
                          name: Optional[str] = None,
                          start_condition: str = '',
                          hold_condition: str = '',
                          end_condition: str = ''):
        """
        Positions the gripper in contact with the door before pushing to open.
        : param root_link: root link of the kinematic chain
        : param tip_link: end effector
        : param door object: name of the object to be pushed
        : param door_handle: name of the object handle
        : param root_V_object_rotation_axis: door rotation axis w.r.t root
        : param root_V_object_normal: door normal w.r.t root
        """
        self.add_motion_goal(motion_goal_class=PrePushDoor.__name__,
                             root_link=root_link,
                             tip_link=tip_link,
                             door_object=door_object,
                             door_handle=door_handle,
                             tip_group=tip_group,
                             root_group=root_group,
                             weight=weight,
                             name=name,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_diff_drive_base(self,
                            goal_pose: PoseStamped,
                            tip_link: str,
                            root_link: str,
                            tip_group: Optional[str] = None,
                            root_group: Optional[str] = None,
                            reference_linear_velocity: Optional[float] = None,
                            reference_angular_velocity: Optional[float] = None,
                            weight: Optional[float] = None,
                            name: Optional[str] = None,
                            start_condition: str = '',
                            hold_condition: str = '',
                            end_condition: str = '',
                            **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        It is specifically for differential drives. Will drive towards the goal the following way:
        1. orient to goal
        2. drive to goal position in a straight line
        3. orient to goal orientation
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        """
        self.add_motion_goal(motion_goal_class=DiffDriveBaseGoal.__name__,
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             root_link=root_link,
                             root_group=root_group,
                             tip_group=tip_group,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_grasp_bar(self,
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
                      weight: Optional[float] = None,
                      name: Optional[str] = None,
                      start_condition: str = '',
                      hold_condition: str = '',
                      end_condition: str = '',
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
        """
        self.add_motion_goal(motion_goal_class=GraspBar.__name__,
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
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_limit_cartesian_velocity(self,
                                     tip_link: str,
                                     root_link: str,
                                     tip_group: Optional[str] = None,
                                     root_group: Optional[str] = None,
                                     max_linear_velocity: float = 0.1,
                                     max_angular_velocity: float = 0.5,
                                     weight: Optional[float] = None,
                                     hard: bool = False,
                                     name: Optional[str] = None,
                                     start_condition: str = '',
                                     hold_condition: str = '',
                                     end_condition: str = '',
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
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        self.add_motion_goal(motion_goal_class=CartesianVelocityLimit.__name__,
                             root_link=root_link,
                             root_group=root_group,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             weight=weight,
                             max_linear_velocity=max_linear_velocity,
                             max_angular_velocity=max_angular_velocity,
                             hard=hard,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_pointing(self,
                     goal_point: PointStamped,
                     tip_link: str,
                     pointing_axis: Vector3Stamped,
                     root_link: str,
                     tip_group: Optional[str] = None,
                     root_group: Optional[str] = None,
                     max_velocity: float = 0.3,
                     weight: Optional[float] = None,
                     name: Optional[str] = None,
                     start_condition: str = '',
                     hold_condition: str = '',
                     end_condition: str = '',
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
        """
        self.add_motion_goal(motion_goal_class=Pointing.__name__,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             goal_point=goal_point,
                             root_link=root_link,
                             root_group=root_group,
                             pointing_axis=pointing_axis,
                             max_velocity=max_velocity,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_real_time_pointing(self,
                               tip_link: str,
                               pointing_axis: Vector3Stamped,
                               root_link: str,
                               topic_name: str,
                               tip_group: Optional[str] = None,
                               root_group: Optional[str] = None,
                               max_velocity: float = 0.3,
                               weight: Optional[float] = None,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               hold_condition: str = '',
                               end_condition: str = '',
                               **kwargs: goal_parameter):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param topic_name: name of a topic of type PointStamped
        :param root_link: root link of the kinematic chain.
        :param tip_group: if tip_link is not unique, search this group for matches.
        :param root_group: if root_link is not unique, search this group for matches.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        """
        self.add_motion_goal(motion_goal_class=RealTimePointing.__name__,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             root_link=root_link,
                             topic_name=topic_name,
                             root_group=root_group,
                             pointing_axis=pointing_axis,
                             max_velocity=max_velocity,
                             weight=weight,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def set_prediction_horizon(self, prediction_horizon: int, **kwargs: goal_parameter):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        self.add_motion_goal(motion_goal_class=SetPredictionHorizon.__name__,
                             prediction_horizon=prediction_horizon,
                             **kwargs)

    def add_carry_my_luggage(self,
                             name: str,
                             tracked_human_position_topic_name: str = '/robokudovanessa/human_position',
                             laser_topic_name: str = '/hsrb/base_scan',
                             point_cloud_laser_topic_name: Optional[str] = None,
                             odom_joint_name: str = 'brumbrum',
                             root_link: Optional[str] = None,
                             camera_link: str = 'head_rgbd_sensor_link',
                             distance_to_target_stop_threshold: float = 1,
                             laser_scan_age_threshold: float = 2,
                             laser_distance_threshold: float = 0.5,
                             laser_distance_threshold_width: float = 0.8,
                             laser_avoidance_angle_cutout: float = np.pi / 4,
                             laser_avoidance_sideways_buffer: float = 0.04,
                             base_orientation_threshold: float = np.pi / 16,
                             tracked_human_position_topic_name_timeout: int = 30,
                             max_rotation_velocity: float = 0.5,
                             max_rotation_velocity_head: float = 1,
                             max_translation_velocity: float = 0.38,
                             traj_tracking_radius: float = 0.4,
                             height_for_camera_target: float = 1,
                             laser_frame_id: str = 'base_range_sensor_link',
                             target_age_threshold: float = 2,
                             target_age_exception_threshold: float = 5,
                             clear_path: bool = False,
                             drive_back: bool = False,
                             enable_laser_avoidance: bool = True,
                             start_condition: str = '',
                             hold_condition: str = '',
                             end_condition: str = ''):
        """
        :param name: name of the goal
        :param tracked_human_position_topic_name: name of the topic where the tracked human is published
        :param laser_topic_name: topic name of the laser scanner
        :param point_cloud_laser_topic_name: topic name of a second laser scanner, e.g. from a point cloud to laser scanner node
        :param odom_joint_name: name of the odom joint
        :param root_link: will use global reference frame
        :param camera_link: link of the camera that will point to the tracked human
        :param distance_to_target_stop_threshold: will pause if closer than this many meter to the target
        :param laser_scan_age_threshold: giskard will complain if scans are older than this many seconds
        :param laser_distance_threshold: this and width are used to crate a stopping zone around the robot.
                                            laser distance draws a circle around the robot and width lines to the left and right.
                                            the stopping zone is the minimum of the two.
        :param laser_distance_threshold_width: see laser_distance_threshold
        :param laser_avoidance_angle_cutout: if something is in the stop zone in front of the robot in +/- this angle range
                                                giskard will pause, otherwise it will try to dodge left or right
        :param laser_avoidance_sideways_buffer: increase this if the robot is shaking too much if something is to its
                                                left and right at the same time.
        :param base_orientation_threshold: giskard will align the base of the robot to the target, this is a +/- buffer to avoid shaking
        :param tracked_human_position_topic_name_timeout: on start up, wait this long for tracking msg to arrive
        :param max_rotation_velocity: how quickly the base can change orientation
        :param max_rotation_velocity_head: how quickly the head rotates
        :param max_translation_velocity: how quickly the base drives
        :param traj_tracking_radius: how close the robots root link will try to stick to the path in meter
        :param height_for_camera_target: target tracking with head will ignore the published height, but use this instead
        :param laser_frame_id: frame_id of the laser scanner
        :param target_age_threshold: will stop looking at the target if the messages are older than this many seconds
        :param target_age_exception_threshold: if there are no messages from the tracked_human_position_topic_name
                                                            topic for this many seconds, cancel
        :param clear_path: clear the saved path. if called repeated will, giskard would just continue the old path if not cleared
        :param drive_back: follow the saved path to drive back
        :param enable_laser_avoidance:
        :param start_condition:
        :param hold_condition:
        :param end_condition:
        """
        self.add_motion_goal(motion_goal_class=CarryMyBullshit.__name__,
                             name=name,
                             patrick_topic_name=tracked_human_position_topic_name,
                             laser_topic_name=laser_topic_name,
                             point_cloud_laser_topic_name=point_cloud_laser_topic_name,
                             odom_joint_name=odom_joint_name,
                             root_link=root_link,
                             camera_link=camera_link,
                             distance_to_target_stop_threshold=distance_to_target_stop_threshold,
                             laser_scan_age_threshold=laser_scan_age_threshold,
                             laser_distance_threshold=laser_distance_threshold,
                             laser_distance_threshold_width=laser_distance_threshold_width,
                             laser_avoidance_angle_cutout=laser_avoidance_angle_cutout,
                             laser_avoidance_sideways_buffer=laser_avoidance_sideways_buffer,
                             base_orientation_threshold=base_orientation_threshold,
                             wait_for_patrick_timeout=tracked_human_position_topic_name_timeout,
                             max_rotation_velocity=max_rotation_velocity,
                             max_rotation_velocity_head=max_rotation_velocity_head,
                             max_translation_velocity=max_translation_velocity,
                             traj_tracking_radius=traj_tracking_radius,
                             height_for_camera_target=height_for_camera_target,
                             laser_frame_id=laser_frame_id,
                             target_age_threshold=target_age_threshold,
                             target_age_exception_threshold=target_age_exception_threshold,
                             clear_path=clear_path,
                             drive_back=drive_back,
                             enable_laser_avoidance=enable_laser_avoidance,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition)

    def add_cartesian_orientation(self,
                                  goal_orientation: QuaternionStamped,
                                  tip_link: str,
                                  root_link: str,
                                  tip_group: Optional[str] = None,
                                  root_group: Optional[str] = None,
                                  reference_velocity: Optional[float] = None,
                                  weight: Optional[float] = None,
                                  absolute: bool = False,
                                  name: Optional[str] = None,
                                  start_condition: str = '',
                                  hold_condition: str = '',
                                  end_condition: str = '',
                                  **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_orientation.
        :param goal_orientation:
        :param tip_link: tip link of kinematic chain
        :param root_link: root link of kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: rad/s, approx limit
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        """
        self.add_motion_goal(motion_goal_class=CartesianOrientation.__name__,
                             goal_orientation=goal_orientation,
                             tip_link=tip_link,
                             root_link=root_link,
                             tip_group=tip_group,
                             root_group=root_group,
                             reference_velocity=reference_velocity,
                             weight=weight,
                             absolute=absolute,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def set_seed_configuration(self,
                               seed_configuration: Dict[str, float],
                               group_name: Optional[str] = None,
                               name: Optional[str] = None):
        """
        Only meant for use with projection. Changes the world state to seed_configuration before starting planning,
        without having to plan a motion to it like with add_joint_position
        """
        self.add_motion_goal(motion_goal_class=SetSeedConfiguration.__name__,
                             seed_configuration=seed_configuration,
                             group_name=group_name,
                             name=name)

    def set_seed_odometry(self,
                          base_pose: PoseStamped,
                          group_name: Optional[str] = None,
                          name: Optional[str] = None):
        """
        Only meant for use with projection. Overwrites the odometry transform with base_pose.
        """
        self.add_motion_goal(motion_goal_class=SetOdometry.__name__,
                             group_name=group_name,
                             base_pose=base_pose,
                             name=name)

    def add_cartesian_pose_straight(self,
                                    goal_pose: PoseStamped,
                                    tip_link: str,
                                    root_link: str,
                                    tip_group: Optional[str] = None,
                                    root_group: Optional[str] = None,
                                    reference_linear_velocity: Optional[float] = None,
                                    reference_angular_velocity: Optional[float] = None,
                                    weight: Optional[float] = None,
                                    absolute: bool = False,
                                    name: Optional[str] = None,
                                    start_condition: str = '',
                                    hold_condition: str = '',
                                    end_condition: str = '',
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
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        """
        self.add_motion_goal(motion_goal_class=CartesianPoseStraight.__name__,
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             root_link=root_link,
                             root_group=root_group,
                             weight=weight,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             name=name,
                             absolute=absolute,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_cartesian_position(self,
                               goal_point: PointStamped,
                               tip_link: str,
                               root_link: str,
                               tip_group: Optional[str] = None,
                               root_group: Optional[str] = None,
                               reference_velocity: Optional[float] = 0.2,
                               weight: Optional[float] = None,
                               absolute: bool = False,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               hold_condition: str = '',
                               end_condition: str = '',
                               **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_point.
        :param goal_point:
        :param tip_link: tip link of the kinematic chain
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: m/s
        :param weight:
        :param absolute: if False, the goal pose is reevaluated if start_condition turns True.
        """
        self.add_motion_goal(motion_goal_class=CartesianPosition.__name__,
                             goal_point=goal_point,
                             tip_link=tip_link,
                             root_link=root_link,
                             tip_group=tip_group,
                             root_group=root_group,
                             reference_velocity=reference_velocity,
                             weight=weight,
                             absolute=absolute,
                             name=name,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)

    def add_cartesian_position_straight(self,
                                        goal_point: PointStamped,
                                        tip_link: str,
                                        root_link: str,
                                        tip_group: Optional[str] = None,
                                        root_group: Optional[str] = None,
                                        reference_velocity: float = None,
                                        weight: Optional[float] = None,
                                        absolute: bool = False,
                                        name: Optional[str] = None,
                                        start_condition: str = '',
                                        hold_condition: str = '',
                                        end_condition: str = '',
                                        **kwargs: goal_parameter):
        """
        Same as set_translation_goal, but will try to move in a straight line.
        """
        self.add_motion_goal(motion_goal_class=CartesianPositionStraight.__name__,
                             goal_point=goal_point,
                             tip_link=tip_link,
                             root_link=root_link,
                             tip_group=tip_group,
                             root_group=root_group,
                             reference_velocity=reference_velocity,
                             weight=weight,
                             name=name,
                             absolute=absolute,
                             start_condition=start_condition,
                             hold_condition=hold_condition,
                             end_condition=end_condition,
                             **kwargs)


class MonitorWrapper:
    _monitors: List[Monitor]
    max_trajectory_length_set: bool
    avoid_name_conflict: bool

    def __init__(self, robot_name: str, avoid_name_conflict: bool = False):
        self._robot_name = robot_name
        self.avoid_name_conflict = avoid_name_conflict
        self.max_trajectory_length_set = False
        self.reset()

    def get_monitors(self) -> List[Monitor]:
        return self._monitors

    def get_anded_monitor_names(self) -> str:
        non_cancel_monitors = []
        for monitor in self._monitors:
            if monitor.monitor_class not in get_all_classes_in_package('giskardpy.monitors', CancelMotion):
                non_cancel_monitors.append(f'\'{monitor.name}\'')
        return ' and '.join(non_cancel_monitors)

    def reset(self):
        self._monitors = []

    def add_monitor(self, *,
                    monitor_class: str,
                    name: Optional[str] = None,
                    start_condition: str = '',
                    **kwargs) -> str:
        """
        Generic function to add a monitor.
        :param monitor_class: Name of a class defined in src/giskardpy/monitors
        :param name: a unique name for the goal, will use class name by default
        :param start_condition: a logical expression to define the start condition for this monitor. e.g.
                                    not 'monitor1' and ('monitor2' or 'monitor3')
        :param kwargs: kwargs for __init__ function of motion_goal_class
        :return: the name of the monitor with added quotes to be used in logical expressions for conditions.
        """
        name = name or monitor_class
        if self.avoid_name_conflict:
            name = f'M{str(len(self._monitors))} {name}'
        if [x for x in self._monitors if x.name == name]:
            raise KeyError(f'monitor named {name} already exists.')
        monitor = giskard_msgs.Monitor()
        monitor.name = name
        monitor.monitor_class = monitor_class
        monitor.start_condition = start_condition
        monitor.kwargs = kwargs_to_json(kwargs)
        self._monitors.append(monitor)
        if not name.startswith('\'') and not name.startswith('"'):
            name = f'\'{name}\''  # put all monitor names in quotes so that the user doesn't have to
        return name

    def add_local_minimum_reached(self,
                                  name: Optional[str] = None,
                                  stay_true: bool = True,
                                  start_condition: str = ''):
        """
        True if the world is currently in a local minimum.
        """
        return self.add_monitor(monitor_class=LocalMinimumReached.__name__,
                                name=name,
                                start_condition=start_condition,
                                stay_true=stay_true)

    def add_time_above(self,
                       threshold: float,
                       name: Optional[str] = None,
                       start_condition: str = ''):
        """
        True if the length of the trajectory is above threshold
        """
        return self.add_monitor(monitor_class=TimeAbove.__name__,
                                name=name,
                                start_condition=start_condition,
                                threshold=threshold)

    def add_joint_position(self,
                           goal_state: Dict[str, float],
                           threshold: float = 0.01,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           stay_true: bool = True) -> str:
        """
        True if all joints in goal_state are closer than threshold to their respective value.
        """
        return self.add_monitor(monitor_class=JointGoalReached.__name__,
                                name=name,
                                goal_state=goal_state,
                                threshold=threshold,
                                start_condition=start_condition,
                                stay_true=stay_true)

    def add_cartesian_pose(self,
                           root_link: str,
                           tip_link: str,
                           goal_pose: PoseStamped,
                           root_group: Optional[str] = None,
                           tip_group: Optional[str] = None,
                           position_threshold: float = 0.01,
                           orientation_threshold: float = 0.01,
                           absolute: bool = False,
                           name: Optional[str] = None,
                           start_condition: str = '',
                           stay_true: bool = True):
        """
        True if tip_link is closer than the thresholds to goal_pose.
        """
        return self.add_monitor(monitor_class=PoseReached.__name__,
                                name=name,
                                root_link=root_link,
                                tip_link=tip_link,
                                goal_pose=goal_pose,
                                root_group=root_group,
                                tip_group=tip_group,
                                absolute=absolute,
                                start_condition=start_condition,
                                position_threshold=position_threshold,
                                orientation_threshold=orientation_threshold,
                                stay_true=stay_true)

    def add_cartesian_position(self,
                               root_link: str,
                               tip_link: str,
                               goal_point: PointStamped,
                               root_group: Optional[str] = None,
                               tip_group: Optional[str] = None,
                               threshold: float = 0.01,
                               absolute: bool = False,
                               name: Optional[str] = None,
                               start_condition: str = '',
                               stay_true: bool = True) -> str:
        """
        True if tip_link is closer than threshold to goal_point.
        """
        return self.add_monitor(monitor_class=PositionReached.__name__,
                                name=name,
                                root_link=root_link,
                                tip_link=tip_link,
                                goal_point=goal_point,
                                root_group=root_group,
                                start_condition=start_condition,
                                absolute=absolute,
                                tip_group=tip_group,
                                threshold=threshold,
                                stay_true=stay_true)

    def add_distance_to_line(self,
                             root_link: str,
                             tip_link: str,
                             center_point: PointStamped,
                             line_axis: Vector3Stamped,
                             line_length: float,
                             name: Optional[str] = None,
                             root_group: Optional[str] = None,
                             tip_group: Optional[str] = None,
                             stay_true: bool = True,
                             start_condition: str = '',
                             threshold: float = 0.01):
        """
        True if tip_link is closer than threshold to the line defined by center_point, line_axis and line_length.
        """
        return self.add_monitor(monitor_class=DistanceToLine.__name__,
                                name=name,
                                center_point=center_point,
                                line_axis=line_axis,
                                line_length=line_length,
                                root_link=root_link,
                                tip_link=tip_link,
                                start_condition=start_condition,
                                root_group=root_group,
                                stay_true=stay_true,
                                tip_group=tip_group,
                                threshold=threshold)

    def add_cartesian_orientation(self,
                                  root_link: str,
                                  tip_link: str,
                                  goal_orientation: QuaternionStamped,
                                  root_group: Optional[str] = None,
                                  tip_group: Optional[str] = None,
                                  threshold: float = 0.01,
                                  absolute: bool = False,
                                  name: Optional[str] = None,
                                  start_condition: str = '',
                                  stay_true: bool = True):
        """
        True if tip_link is closer than threshold to goal_orientation
        """
        return self.add_monitor(monitor_class=OrientationReached.__name__,
                                name=name,
                                root_link=root_link,
                                tip_link=tip_link,
                                goal_orientation=goal_orientation,
                                root_group=root_group,
                                tip_group=tip_group,
                                absolute=absolute,
                                start_condition=start_condition,
                                threshold=threshold,
                                stay_true=stay_true)

    def add_pointing_at(self,
                        goal_point: PointStamped,
                        tip_link: str,
                        pointing_axis: Vector3Stamped,
                        root_link: str,
                        name: Optional[str] = None,
                        tip_group: Optional[str] = None,
                        start_condition: str = '',
                        root_group: Optional[str] = None,
                        threshold: float = 0.01) -> str:
        """
        True if pointing_axis of tip_link is pointing at goal_point withing threshold.
        """
        return self.add_monitor(monitor_class=PointingAt.__name__,
                                name=name,
                                tip_link=tip_link,
                                goal_point=goal_point,
                                root_link=root_link,
                                tip_group=tip_group,
                                start_condition=start_condition,
                                root_group=root_group,
                                pointing_axis=pointing_axis,
                                threshold=threshold)

    def add_vectors_aligned(self,
                            root_link: str,
                            tip_link: str,
                            goal_normal: Vector3Stamped,
                            tip_normal: Vector3Stamped,
                            name: Optional[str] = None,
                            start_condition: str = '',
                            root_group: Optional[str] = None,
                            tip_group: Optional[str] = None,
                            threshold: float = 0.01) -> str:
        """
        True if tip_normal of tip_link is aligned with goal_normal within threshold.
        """
        return self.add_monitor(monitor_class=VectorsAligned.__name__,
                                name=name,
                                root_link=root_link,
                                tip_link=tip_link,
                                goal_normal=goal_normal,
                                tip_normal=tip_normal,
                                start_condition=start_condition,
                                root_group=root_group,
                                tip_group=tip_group,
                                threshold=threshold)

    def add_end_motion(self,
                       start_condition: str,
                       name: Optional[str] = None) -> str:
        """
        Ends the motion execution/planning if all start_condition are True.
        Use this to describe when your motion should end.
        """
        return self.add_monitor(monitor_class=EndMotion.__name__,
                                name=name,
                                start_condition=start_condition)

    def add_cancel_motion(self,
                          start_condition: str,
                          error_message: str,
                          error_code: int = GiskardError.ERROR,
                          name: Optional[str] = None) -> str:
        """
        Cancels the motion if all start_condition are True and will make Giskard return the specified error code.
        Use this to describe when failure conditions.
        """
        return self.add_monitor(monitor_class=CancelMotion.__name__,
                                name=name,
                                start_condition=start_condition,
                                error_message=error_message,
                                error_code=error_code)

    def add_max_trajectory_length(self,
                                  max_trajectory_length: Optional[float] = None) -> str:
        """
        A monitor that cancels the motion if the trajectory is longer than max_trajectory_length.
        """
        self.max_trajectory_length_set = True
        return self.add_monitor(name=None,
                                monitor_class=SetMaxTrajectoryLength.__name__,
                                new_length=max_trajectory_length,
                                start_condition='')

    def add_print(self,
                  message: str,
                  start_condition: str,
                  name: Optional[str] = None) -> str:
        """
        Debugging Monitor.
        Print a message to the terminal if all start_condition are True.
        """
        return self.add_monitor(monitor_class=Print.__name__,
                                name=name,
                                message=message,
                                start_condition=start_condition)

    def add_sleep(self,
                  seconds: float,
                  start_condition: str = '',
                  name: Optional[str] = None) -> str:
        """
        Calls rospy.sleep(seconds) when start_condition are True and turns True itself afterward.
        """
        return self.add_monitor(monitor_class=Sleep.__name__,
                                name=name,
                                seconds=seconds,
                                start_condition=start_condition)

    def add_alternator(self,
                       start_condition: str = '',
                       name: Optional[str] = None,
                       mod: int = 2) -> str:
        """
        Testing monitor.
        True if floor(trajectory_length) % mod == 0.
        """
        if name is None:
            name = Alternator.__name__ + f' % {mod}'
        return self.add_monitor(monitor_class=Alternator.__name__,
                                name=name,
                                start_condition=start_condition,
                                mod=mod)

    def add_payload_alternator(self,
                               start_condition: str = '',
                               name: Optional[str] = None,
                               mod: int = 2) -> str:
        """
        Testing monitor.
        Like add_alternator but as a PayloadMonitor.
        """
        if name is None:
            name = PayloadAlternator.__name__ + f' % {mod}'
        return self.add_monitor(monitor_class=Alternator.__name__,
                                name=name,
                                start_condition=start_condition,
                                mod=mod)


class GiskardWrapper:
    last_feedback: MoveFeedback = None

    def __init__(self, node_name: str = 'giskard', avoid_name_conflict: bool = False):
        """
        Python wrapper for the ROS interface of Giskard.
        :param node_name: node name of Giskard
        :param avoid_name_conflict: if True, Giskard will automatically add an id to monitors and goals to avoid name
                                    conflicts.
        """
        self.world = WorldWrapper(node_name)
        self.monitors = MonitorWrapper(self.robot_name, avoid_name_conflict=avoid_name_conflict)
        self.motion_goals = MotionGoalWrapper(self.robot_name, avoid_name_conflict=avoid_name_conflict)
        self.clear_motion_goals_and_monitors()
        giskard_topic = f'{node_name}/command'
        self._client = SimpleActionClient(giskard_topic, MoveAction)
        self._client.wait_for_server()
        self.clear_motion_goals_and_monitors()
        rospy.sleep(.3)

    def set_avoid_name_conflict(self, value: bool):
        self.avoid_name_conflict = value
        self.monitors.avoid_name_conflict = value
        self.motion_goals.avoid_name_conflict = value

    def add_default_end_motion_conditions(self) -> None:
        """
        1. Adds a local minimum reached monitor and adds it as an end_condition to all previously defined motion goals.
        2. Adds an end motion monitor, start_condition = all previously defined monitors are True.
        3. Adds a cancel motion monitor, start_condition = local minimum reached mit not all other monitors are True.
        4. Adds a max trajectory length monitor, if one wasn't added already.
        """
        local_min_reached_monitor_name = self.monitors.add_local_minimum_reached()
        for goal in self.motion_goals._goals:
            if goal.end_condition:
                goal.end_condition = f'({goal.end_condition}) and {local_min_reached_monitor_name}'
            else:
                goal.end_condition = local_min_reached_monitor_name
        self.monitors.add_end_motion(start_condition=self.monitors.get_anded_monitor_names())
        self.monitors.add_cancel_motion(start_condition=local_min_reached_monitor_name,
                                        error_message=f'local minimum reached',
                                        error_code=GiskardError.LOCAL_MINIMUM)
        if not self.monitors.max_trajectory_length_set:
            self.monitors.add_max_trajectory_length()
        self.monitors.max_trajectory_length_set = False

    @property
    def robot_name(self):
        return self.world.robot_name

    def clear_motion_goals_and_monitors(self):
        """
        Removes all move commands from the current goal, collision entries are left untouched.
        """
        self.motion_goals.reset()
        self.monitors.reset()

    def execute(self, wait: bool = True) -> MoveResult:
        """
        :param wait: this function blocks if wait=True
        :return: result from giskard
        """
        return self._send_action_goal(MoveGoal.EXECUTE, wait)

    def projection(self, wait: bool = True) -> MoveResult:
        """
        Plans, but doesn't execute the goal. Useful, if you just want to look at the planning ghost.
        :param wait: this function blocks if wait=True
        :return: result from Giskard
        """
        return self._send_action_goal(MoveGoal.PROJECTION, wait)

    def _send_action_goal(self, goal_type: int, wait: bool = True) -> Optional[MoveResult]:
        """
        Send goal to Giskard. Use this if you want to specify the goal_type, otherwise stick to wrappers like
        plan_and_execute.
        :param goal_type: one of the constants in MoveGoal
        :param wait: blocks if wait=True
        :return: result from Giskard
        """
        goal = self._create_action_goal()
        goal.type = goal_type
        if wait:
            self._client.send_goal_and_wait(goal)
            return self._client.get_result()
        else:
            self._client.send_goal(goal, feedback_cb=self._feedback_cb)

    def _create_action_goal(self) -> MoveGoal:
        action_goal = MoveGoal()
        action_goal.monitors = self.monitors.get_monitors()
        action_goal.goals = self.motion_goals.get_goals()
        self.clear_motion_goals_and_monitors()
        return action_goal

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

    def _feedback_cb(self, msg: MoveFeedback):
        self.last_feedback = msg
