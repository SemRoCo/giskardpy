import json
from typing import Dict, Tuple, Optional, Union, List

import rospy
from actionlib import SimpleActionClient
from genpy import Message
from rospy import ServiceException
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped
from giskard_msgs.srv import DyeGroupRequest, DyeGroup
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import MarkerArray

from giskard_msgs.msg import MoveAction, MoveGoal, WorldBody, CollisionEntry, MoveResult, Constraint, \
    MoveCmd, MoveFeedback
from giskard_msgs.srv import GetGroupNamesResponse, GetGroupInfoResponse, RegisterGroupRequest
from giskard_msgs.srv import RegisterGroupResponse
from giskard_msgs.srv import UpdateWorld, UpdateWorldRequest, UpdateWorldResponse, GetGroupInfo, \
    GetGroupNames, RegisterGroup
from giskardpy import identifier
from giskardpy.exceptions import DuplicateNameException, UnknownGroupException
from giskardpy.goals.goal import WEIGHT_ABOVE_CA
from giskardpy.god_map import GodMap
from giskardpy.model.utils import make_world_body_box
from giskardpy.model.world import WorldTree
from giskardpy.my_types import goal_parameter
from giskardpy.utils.utils import position_dict_to_joint_states, convert_ros_message_to_dictionary


class GiskardWrapper(object):
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
        self._god_map = GodMap.init_from_paramserver(node_name, upload_config=False)
        self._world = WorldTree(self._god_map)
        self._world.delete_all_but_robot()
        self.collisions = []
        self.clear_cmds()
        self._object_js_topics = {}
        rospy.sleep(.3)

    def register_group(self, group_name: str, parent_group_name: str, root_link_name: str):
        req = RegisterGroupRequest()
        req.group_name = group_name
        req.parent_group_name = parent_group_name
        req.root_link_name = root_link_name
        res = self._register_groups_srv.call(req)  # type: RegisterGroupResponse
        if res.error_codes == res.DUPLICATE_GROUP_ERROR:
            raise DuplicateNameException(f'Group with name {group_name} already exists.')
        if res.error_codes == res.BUSY:
            raise ServiceException('Giskard is busy and can\'t process service call.')

    def _feedback_cb(self, msg: MoveFeedback):
        self.last_feedback = msg

    def get_robot_name(self) -> str:
        return self._god_map.unsafe_get_data(identifier.robot_group_name)

    def get_robot_root_link(self) -> str:
        """
        Returns the name of the robot's root link
        """
        return str(self._world.groups[self.get_robot_name()].root_link_name)

    def set_cart_goal(self,
                      goal_pose: PoseStamped,
                      tip_link: str,
                      root_link: str,
                      weight: Optional[float] = None,
                      max_linear_velocity: Optional[float] = None,
                      max_angular_velocity: Optional[float] = None,
                      **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param max_linear_velocity: m/s, default 0.1
        :param max_angular_velocity: rad/s, default 0.5
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.set_json_goal(constraint_type='CartesianPose',
                           goal_pose=goal_pose,
                           tip_link=tip_link,
                           root_link=root_link,
                           weight=weight,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           **kwargs)

    def set_straight_cart_goal(self,
                               goal_pose: PoseStamped,
                               tip_link: str,
                               root_link: str,
                               weight: Optional[float] = None,
                               max_linear_velocity: Optional[float] = None,
                               max_angular_velocity: Optional[float] = None,
                               **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link on the straightest
        line into the goal pose
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param max_linear_velocity: m/s, default 0.1
        :param max_angular_velocity: rad/s, default 0.5
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.set_json_goal(constraint_type='CartesianPoseStraight',
                           goal_pose=goal_pose,
                           tip_link=tip_link,
                           root_link=root_link,
                           weight=weight,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           **kwargs)

    def set_translation_goal(self,
                             goal_point: PointStamped,
                             tip_link: str,
                             root_link: str,
                             weight: Optional[float] = None,
                             max_velocity: Optional[float] = None,
                             **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal position
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_point: the goal pose, orientation will be ignored
        :param max_velocity: m/s, default 0.1
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.set_json_goal(constraint_type='CartesianPosition',
                           goal_point=goal_point,
                           tip_link=tip_link,
                           root_link=root_link,
                           weight=weight,
                           max_velocity=max_velocity,
                           **kwargs)

    def set_straight_translation_goal(self,
                                      goal_pose: PoseStamped,
                                      tip_link: str,
                                      root_link: str,
                                      weight: Optional[float] = None,
                                      max_velocity: Optional[float] = None,
                                      **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link on the straightest
        line into the goal position
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose, orientation will be ignored
        :param max_velocity: m/s, default 0.1
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.set_json_goal(constraint_type='CartesianPositionStraight',
                           goal_pose=goal_pose,
                           tip_link=tip_link,
                           root_link=root_link,
                           weight=weight,
                           max_velocity=max_velocity,
                           **kwargs)

    def set_rotation_goal(self,
                          goal_orientation: PoseStamped,
                          tip_link: str,
                          root_link: str,
                          weight: Optional[float] = None,
                          max_velocity: Optional[float] = None,
                          **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal orientation
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_orientation: the goal pose, position will be ignored
        :param max_velocity: rad/s, default 0.5
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.set_json_goal(constraint_type='CartesianOrientation',
                           goal_orientation=goal_orientation,
                           tip_link=tip_link,
                           root_link=root_link,
                           weight=weight,
                           max_velocity=max_velocity,
                           **kwargs)

    def set_joint_goal(self,
                       goal_state: dict,
                       weight: Optional[float] = None,
                       max_velocity: Optional[float] = None,
                       hard: bool = False,
                       **kwargs: goal_parameter):
        """
        This goal will move the robots joint to the desired configuration.
        :param goal_state: Can either be a joint state messages or a dict mapping joint name to position.
        :param weight: default WEIGHT_BELOW_CA
        :param max_velocity: default is the default of the added joint goals
        """
        self.set_json_goal(constraint_type='JointPositionList',
                           goal_state=goal_state,
                           weight=weight,
                           max_velocity=max_velocity,
                           hard=hard,
                           **kwargs)

    def set_align_planes_goal(self,
                              tip_link: str,
                              tip_normal: Vector3Stamped,
                              root_link: Optional[str] = None,
                              root_normal: Optional[Vector3Stamped] = None,
                              max_angular_velocity: Optional[float] = None,
                              weight: Optional[float] = None,
                              **kwargs: goal_parameter):
        """
        This Goal will use the kinematic chain between tip and root normal to align both
        :param root_link: name of the root link for the kinematic chain, default robot root link
        :param tip_link: name of the tip link for the kinematic chain
        :param tip_normal: normal at the tip of the kin chain, default is z axis of robot root link
        :param root_normal: normal at the root of the kin chain
        :param max_angular_velocity: rad/s, default 0.5
        :param weight: default WEIGHT_BELOW_CA
        """
        if root_link is None:
            root_link = self.get_robot_root_link()
        if root_normal is None:
            root_normal = Vector3Stamped()
            root_normal.header.frame_id = self.get_robot_root_link()
            root_normal.vector.z = 1

        self.set_json_goal(constraint_type='AlignPlanes',
                           tip_link=tip_link,
                           tip_normal=tip_normal,
                           root_link=root_link,
                           root_normal=root_normal,
                           max_angular_velocity=max_angular_velocity,
                           weight=weight,
                           **kwargs)

    def set_prediction_horizon(self, prediction_horizon: float, **kwargs: goal_parameter):
        self.set_json_goal(constraint_type='SetPredictionHorizon',
                           prediction_horizon=prediction_horizon,
                           **kwargs)

    def set_limit_cartesian_velocity_goal(self,
                                          root_link: str,
                                          tip_link: str,
                                          weight: Optional[float] = None,
                                          max_linear_velocity: float = 0.1,
                                          max_angular_velocity: float = 0.5,
                                          hard: bool = True,
                                          **kwargs: goal_parameter):
        """
        This goal will limit the cartesian velocity of the tip link relative to root link
        :param root_link: root link of the kin chain
        :param tip_link: tip link of the kin chain
        :param weight: default WEIGHT_ABOVE_CA
        :param max_linear_velocity: m/s, default 0.1
        :param max_angular_velocity: rad/s, default 0.5
        :param hard: default True, will turn this into a hard constraint, that will always be satisfied, can could
                                make some goal combination infeasible
        """
        self.set_json_goal('CartesianVelocityLimit',
                           root_link=root_link,
                           tip_link=tip_link,
                           weight=weight,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           hard=hard,
                           **kwargs)

    def set_grasp_bar_goal(self,
                           root_link: str,
                           tip_link: str,
                           tip_grasp_axis: Vector3Stamped,
                           bar_center: PointStamped,
                           bar_axis: Vector3Stamped,
                           bar_length: float,
                           max_linear_velocity: float = 0.1,
                           max_angular_velocity: float = 0.5,
                           weight: Optional[float] = None,
                           **kwargs: goal_parameter):
        """
        This goal can be used to grasp bars. It's like a cartesian goal with some freedom along one axis.
        :param root_link: root link of the kin chain
        :param tip_link: tip link of the kin chain
        :param tip_grasp_axis: this axis of the tip will be aligned with bar_axis
        :param bar_center: center of the bar
        :param bar_axis: tip_grasp_axis will be aligned with this vector
        :param bar_length: length of the bar
        :param max_linear_velocity: m/s, default 0.1
        :param max_angular_velocity: rad/s, default 0.5
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.set_json_goal(constraint_type='GraspBar',
                           root_link=root_link,
                           tip_link=tip_link,
                           tip_grasp_axis=tip_grasp_axis,
                           bar_center=bar_center,
                           bar_axis=bar_axis,
                           bar_length=bar_length,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           weight=weight,
                           **kwargs)

    def set_overwrite_joint_weights_goal(self, updates: Dict[int, Dict[str, float]], **kwargs: goal_parameter):
        """
        {
            1: {
                'joint1: 0.001,
            }
        }
        """
        self.set_json_goal(constraint_type='OverwriteWeights', updates=updates, **kwargs)

    def set_open_drawer_goal(self, tip_link, object_name_prefix, object_link_name, distance_goal,
                           weight=WEIGHT_ABOVE_CA):
        """
        :type tip_link: str
        :param tip_link: tip of manipulator (gripper) which is used
        :type object_name_prefix: object name link prefix
        :param object_name_prefix: string
        :type object_link_name str
        :param object_link_name name of the object link name
        :type object_link_name str
        :param object_link_name knob to grasp
        :type distance_goal: float
        :param distance_goal: how far to open
        :type weight float
        :param weight Default = WEIGHT_ABOVE_CA
        """
        self.set_json_goal(u'OpenDrawer',
                           tip_link=tip_link,
                           object_name=object_name_prefix,
                           object_link_name=object_link_name,
                           distance_goal=distance_goal,
                           weight=weight
                           )

    def set_pointing_goal(self,
                          tip_link: str,
                          goal_point: PointStamped,
                          root_link: Optional[str] = None,
                          pointing_axis: Optional[Vector3Stamped] = None,
                          weight: Optional[float] = None,
                          **kwargs: goal_parameter):
        """
        Uses the kinematic chain from root_link to tip_link to move the pointing axis, such that it points to the goal point.
        :param tip_link: name of the tip of the kin chain
        :param goal_point: where the pointing_axis will point towards
        :param root_link: name of the root of the kin chain
        :param pointing_axis: default is z axis, this axis will point towards the goal_point
        :param weight: default WEIGHT_BELOW_CA
        """
        if root_link is None:
            root_link = self.get_robot_root_link()
        self.set_json_goal(constraint_type='Pointing',
                           tip_link=tip_link,
                           goal_point=goal_point,
                           root_link=root_link,
                           pointing_axis=pointing_axis,
                           weight=weight,
                           **kwargs)

    def set_json_goal(self,
                      constraint_type: str,
                      **kwargs: goal_parameter):
        """
        Set a goal for any of the goals defined in Constraints.py
        :param constraint_type: Name of the Goal
        :param kwargs: maps constraint parameter names to values. Values should be float, str or ros messages.
        """
        constraint = Constraint()
        constraint.type = constraint_type
        for k, v in kwargs.copy().items():
            if v is None:
                del kwargs[k]
            if isinstance(v, Message):
                kwargs[k] = convert_ros_message_to_dictionary(v)
        constraint.parameter_value_pair = json.dumps(kwargs)
        self.cmd_seq[-1].constraints.append(constraint)

    def _set_collision_entries(self, collisions: List[CollisionEntry]):
        """
        Adds collision entries to the current goal
        :param collisions: list of CollisionEntry
        """
        self.cmd_seq[-1].collisions.extend(collisions)

    def allow_collision(self, group1: str = CollisionEntry.ALL, group2: str = CollisionEntry.ALL):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = str(group1)
        collision_entry.group2 = str(group2)
        self._set_collision_entries([collision_entry])

    def avoid_collision(self,
                        min_distance: float = -1,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL):
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

    def avoid_joint_limits(self,
                           percentage: int = 15,
                           weight: Optional[float] = None):
        """
        This goal will push joints away from their position limits
        :param percentage: default 15, if limits are 0-100, the constraint will push into the 15-85 range
        :param weight: default WEIGHT_BELOW_CA
        """
        self.set_json_goal(constraint_type='AvoidJointLimits',
                           percentage=percentage,
                           weight=weight)

    def allow_self_collision(self):
        """
        Allows the collision with itself for the next goal.
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = self.get_robot_name()
        collision_entry.group2 = self.get_robot_name()
        self._set_collision_entries([collision_entry])

    def avoid_self_collision(self, min_distance: float = -1):
        """
        Avoid collisions with itself for the next goal.
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.group1 = self.get_robot_name()
        collision_entry.group2 = self.get_robot_name()
        collision_entry.distance = min_distance
        self._set_collision_entries([collision_entry])

    def avoid_all_collisions(self, min_distance: float = -1):
        """
        Avoids all collisions for next goal. The distance will override anything from the config file.
        If you don't want to override the distance, don't call this function. Avoid all is the default, if you don't
        add any collision entries.
        :param min_distance: the distance that giskard is trying to keep from all objects
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        self._set_collision_entries([collision_entry])

    def add_cmd(self):
        """
        Adds another command to the goal sequence. Any set goal calls will be added the this new goal.
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
        :param wait: this function block if wait=True
        :return: result from giskard
        """
        return self.send_goal(MoveGoal.PLAN_AND_EXECUTE, wait)

    def check_reachability(self, wait: bool = True) -> MoveResult:
        """
        Not implemented
        :param wait: this function block if wait=True
        :return: result from giskard
        """
        raise NotImplementedError('reachability check is not implemented')

    def plan(self, wait: bool = True) -> MoveResult:
        """
        Plans, but doesn't execute the goal. Useful, if you just want to look at the planning ghost.
        :param wait: this function block if wait=True
        :return: result from giskard
        """
        return self.send_goal(MoveGoal.PLAN_ONLY, wait)

    def send_goal(self, goal_type: str, wait: bool = True) -> Optional[MoveResult]:
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
        Stops any goal that Giskard is processing.
        """
        self._client.cancel_goal()

    def get_result(self, timeout: rospy.Duration = rospy.Duration()) -> MoveResult:
        """
        Waits for giskardpy result and returns it. Only used when plan_and_execute was called with wait=False
        """
        if not self._client.wait_for_result(timeout):
            raise TimeoutError('Timeout while waiting for goal.')
        return self._client.get_result()

    def clear_world(self, timeout: float = 0) -> UpdateWorldResponse:
        """
        Removes any objects and attached objects from Giskard's world and reverts the robots urdf to what it got from
        the parameter server.
        """
        req = UpdateWorldRequest()
        req.operation = UpdateWorldRequest.REMOVE_ALL
        req.timeout = timeout
        return self._update_world_srv.call(req)

    def remove_group(self,
                     name: str,
                     timeout: float = 0) -> UpdateWorldResponse:
        object = WorldBody()
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.REMOVE
        req.timeout = timeout
        req.body = object
        return self._update_world_srv.call(req)

    def add_box(self,
                name: str,
                size: Tuple[float, float, float],
                pose: PoseStamped,
                parent_link: str = '',
                parent_link_group: str = '',
                timeout: float = 1) -> UpdateWorldResponse:
        """
        Adds a new box to the world tree and attaches it to 'parent_link_group'/'parent_link'.
        If 'parent_link_group' and 'parent_link' are empty, the box will be attached to the world root link
        :param name: How the new group will be called
        :param size: X, Y and Z dimensions of the box, respectively
        :param pose: Where the root link of the new object will be positioned
        :param parent_link: Name of the link, the object will get attached to
        :param parent_link_group: Name of the group in which Giskard will serach for 'parent_link'
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
                   timeout: float = 0) -> UpdateWorldResponse:
        """
        If pose is used, frame_id, position and orientation are ignored.
        :param radius: in m
        """
        object = WorldBody()
        object.type = WorldBody.PRIMITIVE_BODY
        object.shape.type = SolidPrimitive.SPHERE
        object.shape.dimensions.append(radius)
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.ADD
        req.timeout = timeout
        req.body = object
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
                 timeout: float = 0) -> UpdateWorldResponse:
        """
        If pose is used, frame_id, position and orientation are ignored.
        :param mesh: path to the meshes location. e.g. package://giskardpy/test/urdfs/meshes/bowl_21.obj
        """
        object = WorldBody()
        object.type = WorldBody.MESH_BODY
        object.mesh = mesh
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.ADD
        req.timeout = timeout
        req.body = object
        req.pose = pose
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
                     timeout: float = 0) -> UpdateWorldResponse:
        """
        If pose is used, frame_id, position and orientation are ignored.
        :param height: in m
        :param radius: in m
        """
        object = WorldBody()
        object.type = WorldBody.PRIMITIVE_BODY
        object.shape.type = SolidPrimitive.CYLINDER
        object.shape.dimensions = [0, 0]
        object.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT] = height
        object.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS] = radius
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.ADD
        req.timeout = timeout
        req.body = object
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        return self._update_world_srv.call(req)

    def update_parent_link_of_group(self,
                                    name: str,
                                    parent_link: str,
                                    parent_link_group: str,
                                    timeout: float = 0) -> UpdateWorldResponse:
        req = UpdateWorldRequest()
        req.operation = UpdateWorldRequest.UPDATE_PARENT_LINK
        req.group_name = str(name)
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        req.timeout = timeout
        return self._update_world_srv.call(req)

    def detach_group(self, object_name: str, timeout: float = 0):
        """
        Detach an object from the robot and add it back to the world.
        Careful though, you could amputate an arm be accident!
        :type object_name: str
        :return: UpdateWorldResponse
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
                 js_topic: str = '',
                 set_js_topic: Optional[str] = None,
                 timeout: float = 0) -> UpdateWorldResponse:
        """
        Adds a urdf to the world
        :param name: name it will have in the world
        :param urdf: urdf as string, no path!
        :param js_topic: Giskard will listen on that topic for joint states and update the urdf accordingly
        :param set_js_topic: A topic that the python wrapper will use to set the urdf joint state.
                                If None, set_js_topic == js_topic
        :return: UpdateWorldResponse
        """
        if set_js_topic is None:
            set_js_topic = js_topic
        urdf_body = WorldBody()
        urdf_body.type = WorldBody.URDF_BODY
        urdf_body.urdf = str(urdf)
        urdf_body.joint_state_topic = str(js_topic)
        req = UpdateWorldRequest()
        req.group_name = str(name)
        req.operation = UpdateWorldRequest.ADD
        req.timeout = timeout
        req.body = urdf_body
        req.pose = pose
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        if js_topic:
            # FIXME publisher has to be removed, when object gets deleted
            # FIXME there could be sync error, if objects get added/removed by something else
            self._object_js_topics[name] = rospy.Publisher(set_js_topic, JointState, queue_size=10)
        return self._update_world_srv.call(req)

    def set_object_joint_state(self, object_name: str, joint_states: Union[JointState, dict]):
        """
        :type object_name: str
        :param joint_states: joint state message or a dict that maps joint name to position
        :return: UpdateWorldResponse
        """
        if isinstance(joint_states, dict):
            joint_states = position_dict_to_joint_states(joint_states)
        self._object_js_topics[object_name].publish(joint_states)

    def dye_group(self, group_name: str, rgba: Tuple[float, float, float, float]):
        req = DyeGroupRequest()
        req.group_name = group_name
        req.color.r = rgba[0]
        req.color.g = rgba[1]
        req.color.b = rgba[2]
        req.color.a = rgba[3]
        return self.dye_group_srv(req)

    def get_group_names(self) -> List[str]:
        """
        returns the names of every object in the world
        """
        resp = self._get_group_names_srv()  # type: GetGroupNamesResponse
        return resp.group_names

    def get_group_info(self, group_name: str) -> GetGroupInfoResponse:
        """
        returns the joint state, joint state topic and pose of the object with the given name
        """
        return self._get_group_info_srv(group_name)

    def get_controlled_joints(self, name: Optional[str] = None):
        if name is None:
            name = self.get_robot_name()
        return self.get_group_info(name).controlled_joints

    def update_group_pose(self, group_name: str, new_pose: PoseStamped, timeout: float = 0) -> UpdateWorldResponse:
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
