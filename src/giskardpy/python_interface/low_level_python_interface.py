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

from giskard_msgs.msg import MoveAction, MoveGoal, WorldBody, CollisionEntry, MoveResult, MoveFeedback, MotionGoal, \
    Monitor
import giskard_msgs.msg as giskard_msgs
from giskard_msgs.srv import DyeGroupRequest, DyeGroup, GetGroupInfoRequest, DyeGroupResponse
from giskard_msgs.srv import GetGroupNamesResponse, GetGroupInfoResponse, RegisterGroupRequest
from giskard_msgs.srv import RegisterGroupResponse
from giskard_msgs.srv import UpdateWorld, UpdateWorldRequest, UpdateWorldResponse, GetGroupInfo, \
    GetGroupNames, RegisterGroup
from giskardpy.exceptions import DuplicateNameException, UnknownGroupException
from giskardpy.goals.monitors.cartesian_monitors import PoseReached, PositionReached, OrientationReached, PointingAt, \
    VectorsAligned, DistanceToLine
from giskardpy.goals.monitors.joint_monitors import JointGoalReached
from giskardpy.goals.monitors.monitors import LocalMinimumReached
from giskardpy.goals.tasks.task import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.god_map import god_map
from giskardpy.model.utils import make_world_body_box
from giskardpy.my_types import goal_parameter
from giskardpy.utils.utils import position_dict_to_joint_states, convert_ros_message_to_dictionary, \
    replace_prefix_name_with_str, kwargs_to_json


class LowLevelGiskardWrapper:
    last_feedback: MoveFeedback = None
    _goals: List[MotionGoal]
    _monitors: List[Monitor]
    _collision_entries: List[CollisionEntry]

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
        self.clear_cmds()
        self._object_js_topics = {}
        rospy.sleep(.3)
        self.robot_name = self.get_group_names()[0]

    # %% action server communication
    def clear_cmds(self):
        """
        Removes all move commands from the current goal, collision entries are left untouched.
        """
        self._goals = []
        self._monitors = []
        self._collision_entries = []

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
        local_min_reached_monitor_name = 'local min reached'
        self.add_monitor(monitor_type=LocalMinimumReached.__name__, monitor_name=local_min_reached_monitor_name)
        for goal in self._goals:
            goal.to_end.append(local_min_reached_monitor_name)
        goal = self._create_action_goal()
        goal.type = goal_type
        if wait:
            self._client.send_goal_and_wait(goal)
            return self._client.get_result()
        else:
            self._client.send_goal(goal, feedback_cb=self._feedback_cb)

    def _create_action_goal(self) -> MoveGoal:
        action_goal = MoveGoal()
        action_goal.monitors = self._monitors
        action_goal.goals = self._goals
        action_goal.collisions = self._collision_entries
        self.clear_cmds()
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

    def add_motion_goal(self, goal_type, goal_name: str = '',
                        to_start: Optional[List[str]] = None,
                        to_hold: Optional[List[str]] = None,
                        to_end: Optional[List[str]] = None,
                        **kwargs):
        motion_goal = MotionGoal()
        motion_goal.name = goal_name
        motion_goal.type = goal_type
        motion_goal.to_start = to_start if to_start is not None else []
        motion_goal.to_hold = to_hold if to_hold is not None else []
        motion_goal.to_end = to_end if to_end is not None else []
        motion_goal.parameter_value_pair = kwargs_to_json(kwargs)
        self._goals.append(motion_goal)

    def add_monitor(self, monitor_type: str, monitor_name: str, **kwargs):
        if [x for x in self._monitors if x.name == monitor_name]:
            raise KeyError(f'monitor named {monitor_name} already exists.')
        monitor = giskard_msgs.Monitor()
        monitor.type = monitor_type
        monitor.name = monitor_name
        monitor.parameter_value_pair = kwargs_to_json(kwargs)
        self._monitors.append(monitor)

    def _feedback_cb(self, msg: MoveFeedback):
        self.last_feedback = msg

    # %% predefined monitors
    def add_joint_position_reached_monitor(self,
                                           goal_state: Dict[str, float],
                                           name: Optional[str] = None,
                                           threshold: float = 0.01,
                                           crucial: bool = True) -> str:
        if name is None:
            name = f'joint position reached {list(goal_state.keys())}'
        self.add_monitor(monitor_type=JointGoalReached.__name__,
                         monitor_name=name,
                         goal_state=goal_state,
                         threshold=threshold,
                         crucial=crucial)
        return name

    def add_cartesian_pose_reached_monitor(self,
                                           name: str,
                                           root_link: str,
                                           tip_link: str,
                                           goal_pose: PoseStamped,
                                           root_group: Optional[str] = None,
                                           tip_group: Optional[str] = None,
                                           position_threshold: float = 0.01,
                                           orientation_threshold: float = 0.01,
                                           crucial: bool = True):
        self.add_monitor(monitor_type=PoseReached.__name__,
                         monitor_name=name,
                         root_link=root_link,
                         tip_link=tip_link,
                         goal_pose=goal_pose,
                         root_group=root_group,
                         tip_group=tip_group,
                         position_threshold=position_threshold,
                         orientation_threshold=orientation_threshold,
                         crucial=crucial)

    def add_cartesian_position_reached_monitor(self,
                                               *,
                                               root_link: str,
                                               tip_link: str,
                                               goal_point: PointStamped,
                                               name: Optional[str] = None,
                                               root_group: Optional[str] = None,
                                               tip_group: Optional[str] = None,
                                               threshold: float = 0.01,
                                               crucial: bool = True) -> str:
        if name is None:
            name = f'{root_link}/{tip_link} position reached'
        self.add_monitor(monitor_type=PositionReached.__name__,
                         monitor_name=name,
                         root_link=root_link,
                         tip_link=tip_link,
                         goal_point=goal_point,
                         root_group=root_group,
                         tip_group=tip_group,
                         threshold=threshold,
                         crucial=crucial)

        return name

    def add_distance_to_line_monitor(self,
                                     *,
                                     root_link: str,
                                     tip_link: str,
                                     center_point: PointStamped,
                                     line_axis: Vector3Stamped,
                                     line_length: float,
                                     name: Optional[str] = None,
                                     root_group: Optional[str] = None,
                                     tip_group: Optional[str] = None,
                                     threshold: float = 0.01,
                                     crucial: bool = True):
        if name is None:
            name = f'{root_link}/{tip_link} distance to line'
        self.add_monitor(monitor_type=DistanceToLine.__name__,
                         monitor_name=name,
                         center_point=center_point,
                         line_axis=line_axis,
                         line_length=line_length,
                         root_link=root_link,
                         tip_link=tip_link,
                         root_group=root_group,
                         tip_group=tip_group,
                         threshold=threshold,
                         crucial=crucial)
        return name

    def add_cartesian_orientation_reached_monitor(self,
                                                  name: str,
                                                  root_link: str,
                                                  tip_link: str,
                                                  goal_orientation: QuaternionStamped,
                                                  root_group: Optional[str] = None,
                                                  tip_group: Optional[str] = None,
                                                  threshold: float = 0.01,
                                                  crucial: bool = True):
        self.add_monitor(monitor_type=OrientationReached.__name__,
                         monitor_name=name,
                         root_link=root_link,
                         tip_link=tip_link,
                         goal_orientation=goal_orientation,
                         root_group=root_group,
                         tip_group=tip_group,
                         threshold=threshold,
                         crucial=crucial)

    def add_pointing_at_monitor(self,
                                goal_point: PointStamped,
                                tip_link: str,
                                pointing_axis: Vector3Stamped,
                                root_link: str,
                                name: Optional[str] = None,
                                tip_group: Optional[str] = None,
                                root_group: Optional[str] = None,
                                threshold: float = 0.01,
                                crucial: bool = True) -> str:
        if name is None:
            name = f'{root_link}/{tip_link} pointing at'
        self.add_monitor(monitor_type=PointingAt.__name__,
                         monitor_name=name,
                         tip_link=tip_link,
                         goal_point=goal_point,
                         root_link=root_link,
                         tip_group=tip_group,
                         root_group=root_group,
                         pointing_axis=pointing_axis,
                         threshold=threshold,
                         crucial=crucial)
        return name

    def add_vectors_aligned_monitor(self,
                                    *,
                                    root_link: str,
                                    tip_link: str,
                                    goal_normal: Vector3Stamped,
                                    tip_normal: Vector3Stamped,
                                    name: Optional[str] = None,
                                    root_group: Optional[str] = None,
                                    tip_group: Optional[str] = None,
                                    threshold: float = 0.01,
                                    crucial: bool = True) -> str:
        if name is None:
            name = f'{root_link}/{tip_link} {goal_normal}/{tip_normal} vectors aligned'
        self.add_monitor(monitor_type=VectorsAligned.__name__,
                         monitor_name=name,
                         root_link=root_link,
                         tip_link=tip_link,
                         goal_normal=goal_normal,
                         tip_normal=tip_normal,
                         root_group=root_group,
                         tip_group=tip_group,
                         threshold=threshold,
                         crucial=crucial)
        return name

    # %% collision avoidance
    def _set_collision_entries(self, collisions: List[CollisionEntry]):
        """
        Adds collision entries to the current goal
        :param collisions: list of CollisionEntry
        """
        self._collision_entries.extend(collisions)

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

    # %% world manipulation
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
        res: RegisterGroupResponse = self._register_groups_srv.call(req)
        if res.error_codes == res.DUPLICATE_GROUP_ERROR:
            raise DuplicateNameException(f'Group with name {new_group_name} already exists.')
        if res.error_codes == res.BUSY:
            raise ServiceException('Giskard is busy and can\'t process service call.')
        return res
