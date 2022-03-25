import json
from typing import Dict, Tuple, Optional, Union

import rospy
from actionlib import SimpleActionClient
from genpy import Message
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PointStamped
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import MarkerArray

from giskard_msgs.msg import MoveAction, MoveGoal, WorldBody, CollisionEntry, MoveResult, Constraint, \
    MoveCmd, MoveFeedback
from giskard_msgs.srv import UpdateWorld, UpdateWorldRequest, UpdateWorldResponse, GetObjectInfo, GetObjectNames, \
    UpdateRvizMarkers, GetAttachedObjects, GetAttachedObjectsResponse, GetObjectNamesResponse
from giskardpy import RobotName
from giskardpy.goals.goal import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
from giskardpy.god_map import GodMap
from giskardpy.model.utils import make_world_body_box
from giskardpy.model.world import WorldTree
from giskardpy.utils.utils import position_dict_to_joint_states, convert_ros_message_to_dictionary


class GiskardWrapper(object):
    last_feedback: MoveFeedback = None

    def __init__(self, node_name='giskard'):
        giskard_topic = '{}/command'.format(node_name)
        if giskard_topic is not None:
            self._client = SimpleActionClient(giskard_topic, MoveAction)
            self._update_world_srv = rospy.ServiceProxy('{}/update_world'.format(node_name), UpdateWorld)
            self._get_object_names_srv = rospy.ServiceProxy('{}/get_object_names'.format(node_name), GetObjectNames)
            self._get_object_info_srv = rospy.ServiceProxy('{}/get_object_info'.format(node_name), GetObjectInfo)
            self._update_rviz_markers_srv = rospy.ServiceProxy('{}/update_rviz_markers'.format(node_name),
                                                               UpdateRvizMarkers)
            self._get_attached_objects_srv = rospy.ServiceProxy('{}/get_attached_objects'.format(node_name),
                                                                GetAttachedObjects)
            self._marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
            rospy.wait_for_service('{}/update_world'.format(node_name))
            self._client.wait_for_server()
        self._god_map = GodMap.init_from_paramserver(node_name, upload_config=False)
        self._world = WorldTree(self._god_map)
        self._world.delete_all_but_robot()
        self.collisions = []
        self.clear_cmds()
        self._object_js_topics = {}
        rospy.sleep(.3)

    def _feedback_cb(self, msg):
        self.last_feedback = msg

    def get_robot_name(self):
        """
        :rtype: str
        """
        return RobotName

    def get_root(self):
        """
        Returns the name of the robot's root link
        :rtype: str
        """
        return str(self._world.groups[RobotName].root_link_name)

    def set_cart_goal(self, goal_pose, tip_link, root_link, max_linear_velocity=None, max_angular_velocity=None,
                      weight=None):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: name of the root link of the kin chain
        :type root_link: str
        :param tip_link: name of the tip link of the kin chain
        :type tip_link: str
        :param goal_pose: the goal pose
        :type goal_pose: PoseStamped
        :param max_linear_velocity: m/s, default 0.1
        :type max_linear_velocity: float
        :param max_angular_velocity: rad/s, default 0.5
        :type max_angular_velocity: float
        :param weight: default WEIGHT_ABOVE_CA
        :type weight: float
        """
        self.set_translation_goal(goal_pose, tip_link, root_link, weight=weight, max_velocity=max_linear_velocity)
        self.set_rotation_goal(goal_pose, tip_link, root_link, weight=weight, max_velocity=max_angular_velocity)

    def set_straight_cart_goal(self, goal_pose, tip_link, root_link, max_linear_velocity=None,
                               max_angular_velocity=None,
                               weight=None):
        """
        This goal will use the kinematic chain between root and tip link to move tip link on the straightest
        line into the goal pose
        :param root_link: name of the root link of the kin chain
        :type root_link: str
        :param tip_link: name of the tip link of the kin chain
        :type tip_link: str
        :param goal_pose: the goal pose
        :type goal_pose: PoseStamped
        :param max_linear_velocity: m/s, default 0.1
        :type max_linear_velocity: float
        :param max_angular_velocity: rad/s, default 0.5
        :type max_angular_velocity: float
        :param weight: default WEIGHT_ABOVE_CA
        :type weight: float
        """
        self.set_straight_translation_goal(goal_pose, tip_link, root_link, max_velocity=max_linear_velocity,
                                           weight=weight)
        self.set_rotation_goal(goal_pose, tip_link, root_link, max_velocity=max_angular_velocity, weight=weight)

    def set_translation_goal(self, goal_pose, tip_link, root_link, weight=None, max_velocity=None, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal position
        :param root_link: name of the root link of the kin chain
        :type root_link: str
        :param tip_link: name of the tip link of the kin chain
        :type tip_link: str
        :param goal_pose: the goal pose, orientation will be ignored
        :type goal_pose: PoseStamped
        :param max_velocity: m/s, default 0.1
        :type max_velocity: float
        :param weight: default WEIGHT_ABOVE_CA
        :type weight: float
        """
        constraint = Constraint()
        constraint.type = 'CartesianPosition'
        params = {'root_link': root_link,
                  'tip_link': tip_link,
                  'goal': convert_ros_message_to_dictionary(goal_pose)}
        if max_velocity:
            params['max_velocity'] = max_velocity
        if weight:
            params['weight'] = weight
        params.update(kwargs)
        constraint.parameter_value_pair = json.dumps(params)
        self.cmd_seq[-1].constraints.append(constraint)

    def set_straight_translation_goal(self, goal_pose, tip_link, root_link, weight=None, max_velocity=None, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link on the straightest
        line into the goal position
        :param root_link: name of the root link of the kin chain
        :type root_link: str
        :param tip_link: name of the tip link of the kin chain
        :type tip_link: str
        :param goal_pose: the goal pose, orientation will be ignored
        :type goal_pose: PoseStamped
        :param max_velocity: m/s, default 0.1
        :type max_velocity: float
        :param weight: default WEIGHT_ABOVE_CA
        :type weight: float
        """
        constraint = Constraint()
        constraint.type = 'CartesianPositionStraight'
        params = {'root_link': root_link,
                  'tip_link': tip_link,
                  'goal': convert_ros_message_to_dictionary(goal_pose)}
        if max_velocity:
            params['max_velocity'] = max_velocity
        if weight:
            params['weight'] = weight
        params.update(kwargs)
        constraint.parameter_value_pair = json.dumps(params)
        self.cmd_seq[-1].constraints.append(constraint)

    def set_rotation_goal(self, goal_pose, tip_link, root_link, weight=None, max_velocity=None, **kwargs):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal orientation
        :param root_link: name of the root link of the kin chain
        :type root_link: str
        :param tip_link: name of the tip link of the kin chain
        :type tip_link: str
        :param goal_pose: the goal pose, position will be ignored
        :type goal_pose: PoseStamped
        :param max_velocity: rad/s, default 0.5
        :type max_velocity: float
        :param weight: default WEIGHT_ABOVE_CA
        :type weight: float
        """
        constraint = Constraint()
        constraint.type = 'CartesianOrientation'
        params = {'root_link': root_link,
                  'tip_link': tip_link,
                  'goal': convert_ros_message_to_dictionary(goal_pose)}
        if max_velocity:
            params['max_velocity'] = max_velocity
        if weight:
            params['weight'] = weight
        params.update(kwargs)
        constraint.parameter_value_pair = json.dumps(params)
        self.cmd_seq[-1].constraints.append(constraint)

    def set_joint_goal(self,
                       goal_state: dict,
                       weight: Optional[float] = None,
                       max_velocity: Optional[float] = None,
                       hard: bool = False):
        """
        This goal will move the robots joint to the desired configuration.
        :param goal_state: Can either be a joint state messages or a dict mapping joint name to position. 
        :param weight: default WEIGHT_BELOW_CA
        :param max_velocity: default is the default of the added joint goals
        """
        constraint = Constraint()
        constraint.type = 'JointPositionList'
        if isinstance(goal_state, JointState):
            goal_state = goal_state
        else:
            goal_state2 = JointState()
            for joint_name, joint_position in goal_state.items():
                goal_state2.name.append(joint_name)
                goal_state2.position.append(joint_position)
            goal_state = goal_state2
        params = {'goal_state': convert_ros_message_to_dictionary(goal_state)}
        if weight is not None:
            params['weight'] = weight
        if max_velocity is not None:
            params['max_velocity'] = max_velocity
        params['hard'] = hard
        constraint.parameter_value_pair = json.dumps(params)
        self.cmd_seq[-1].constraints.append(constraint)

    def set_align_planes_goal(self, tip_link, tip_normal, root_link=None, root_normal=None, max_angular_velocity=None,
                              weight=WEIGHT_ABOVE_CA):
        """
        This Goal will use the kinematic chain between tip and root normal to align both
        :param root_link: name of the root link for the kinematic chain, default robot root link
        :type root_link: str
        :param tip_link: name of the tip link for the kinematic chain
        :type tip_link: str
        :param tip_normal: normal at the tip of the kin chain, default is z axis of robot root link
        :type tip_normal: Vector3Stamped
        :param root_normal: normal at the root of the kin chain
        :type root_normal: Vector3Stamped
        :param max_angular_velocity: rad/s, default 0.5
        :type max_angular_velocity: float
        :param weight: default WEIGHT_BELOW_CA
        :type weight: float
        """
        if root_link is None:
            root_link = self.get_root()
        if root_normal is None:
            root_normal = Vector3Stamped()
            root_normal.header.frame_id = self.get_root()
            root_normal.vector.z = 1

        params = {'tip_link': str(tip_link),
                  'tip_normal': tip_normal,
                  'root_link': str(root_link),
                  'root_normal': root_normal}
        if weight is not None:
            params['weight'] = weight
        if max_angular_velocity is not None:
            params['max_angular_velocity'] = max_angular_velocity
        self.set_json_goal('AlignPlanes', **params)

    def avoid_joint_limits(self, percentage=15, weight=WEIGHT_BELOW_CA):
        """
        This goal will push joints away from their position limits
        :param percentage: default 15, if limits are 0-100, the constraint will push into the 15-85 range
        :type percentage: float
        :param weight: default WEIGHT_BELOW_CA
        :type weight: float
        """
        self.set_json_goal('AvoidJointLimits', percentage=percentage, weight=weight)

    def limit_cartesian_velocity(self, root_link, tip_link, weight=WEIGHT_ABOVE_CA, max_linear_velocity=0.1,
                                 max_angular_velocity=0.5, hard=True):
        """
        This goal will limit the cartesian velocity of the tip link relative to root link
        :param root_link: root link of the kin chain
        :type root_link: str
        :param tip_link: tip link of the kin chain
        :type tip_link: str
        :param weight: default WEIGHT_ABOVE_CA
        :type weight: float
        :param max_linear_velocity: m/s, default 0.1
        :type max_linear_velocity: float
        :param max_angular_velocity: rad/s, default 0.5
        :type max_angular_velocity: float
        :param hard: default True, will turn this into a hard constraint, that will always be satisfied, can could
                                make some goal combination infeasible
        :type hard: bool
        """
        self.set_json_goal('CartesianVelocityLimit',
                           root_link=root_link,
                           tip_link=tip_link,
                           weight=weight,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           hard=hard)

    def grasp_bar(self, root_link, tip_link, tip_grasp_axis, bar_center, bar_axis, bar_length,
                  max_linear_velocity=0.1, max_angular_velocity=0.5, weight=WEIGHT_ABOVE_CA):
        """
        This goal can be used to grasp bars. It's like a cartesian goal with some freedom along one axis.
        :param root_link: root link of the kin chain
        :type root_link: str
        :param tip_link: tip link of the kin chain
        :type tip_link: str
        :param tip_grasp_axis: this axis of the tip will be aligned with bar_axis
        :type tip_grasp_axis: Vector3Stamped
        :param bar_center: center of the bar
        :type bar_center: PointStamped
        :param bar_axis: tip_grasp_axis will be aligned with this vector
        :type bar_axis: Vector3Stamped
        :param bar_length: length of the bar
        :type bar_length: float
        :param max_linear_velocity: m/s, default 0.1
        :type max_linear_velocity: float
        :param max_angular_velocity: rad/s, default 0.5
        :type max_angular_velocity: float
        :param weight: default WEIGHT_ABOVE_CA
        :type weight: float
        """
        self.set_json_goal('GraspBar',
                           root_link=root_link,
                           tip_link=tip_link,
                           tip_grasp_axis=tip_grasp_axis,
                           bar_center=bar_center,
                           bar_axis=bar_axis,
                           bar_length=bar_length,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           weight=weight)

    def overwrite_joint_weights(self, updates: Dict[int, Dict[str, float]]):
        """
        {
            1: {
                'joint1: 0.001,
            }
        }
        """
        self.set_json_goal('OverwriteWeights', updates=updates)

    def set_pointing_goal(self, tip_link, goal_point, root_link=None, pointing_axis=None, weight=None):
        """
        Uses the kinematic chain from root_link to tip_link to move the pointing axis, such that it points to the goal point.
        :param tip_link: name of the tip of the kin chain
        :type tip_link: str
        :param goal_point: where the pointing_axis will point towards
        :type goal_point: PointStamped
        :param root_link: name of the root of the kin chain
        :type root_link: str
        :param pointing_axis: default is z axis, this axis will point towards the goal_point
        :type pointing_axis: Vector3Stamped
        :param weight: default WEIGHT_BELOW_CA
        :type weight: float
        """
        kwargs = {'tip_link': tip_link,
                  'goal_point': goal_point}
        if root_link is not None:
            kwargs['root_link'] = root_link
        else:
            kwargs['root_link'] = self.get_root()
        if pointing_axis is not None:
            kwargs['pointing_axis'] = pointing_axis
        if weight is not None:
            kwargs['weight'] = weight
        kwargs['goal_point'] = goal_point
        self.set_json_goal('Pointing', **kwargs)

    def set_json_goal(self, constraint_type, **kwargs):
        """
        Set a goal for any of the goals defined in Constraints.py
        :param constraint_type: Name of the Goal
        :type constraint_type: str
        :param **kwargs: maps constraint parameter names to values. Values should be float, str or ros messages.
        """
        constraint = Constraint()
        constraint.type = constraint_type
        for k, v in kwargs.items():
            if isinstance(v, Message):
                kwargs[k] = convert_ros_message_to_dictionary(v)
        constraint.parameter_value_pair = json.dumps(kwargs)
        self.cmd_seq[-1].constraints.append(constraint)

    def set_collision_entries(self, collisions):
        """
        Adds collision entries to the current goal
        :param collisions: list of CollisionEntry
        :type collisions: list
        """
        self.cmd_seq[-1].collisions.extend(collisions)

    def allow_collision(self, group1: str = CollisionEntry.ALL, group2: str = CollisionEntry.ALL):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = str(group1)
        collision_entry.group2 = str(group2)
        self.set_collision_entries([collision_entry])

    def avoid_collision(self, min_distance: float, group1: str = CollisionEntry.ALL, group2: str = CollisionEntry.ALL):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        collision_entry.group1 = group1
        collision_entry.group2 = group2
        self.set_collision_entries([collision_entry])

    def allow_all_collisions(self):
        """
        Allows all collisions for next goal.
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        self.set_collision_entries([collision_entry])

    def allow_self_collision(self):
        """
        Allows the collision with itself for the next goal.
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = self.get_robot_name()
        collision_entry.group2 = self.get_robot_name()
        self.set_collision_entries([collision_entry])

    def avoid_self_collision(self, min_distance: float = -1):
        """
        Avoid collisions with itself for the next goal.
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.group1 = self.get_robot_name()
        collision_entry.group2 = self.get_robot_name()
        collision_entry.distance = min_distance
        self.set_collision_entries([collision_entry])

    def avoid_all_collisions(self, min_distance:float = -1):
        """
        Avoids all collisions for next goal. The distance will override anything from the config file.
        If you don't want to override the distance, don't call this function. Avoid all is the default, if you don't
        add any collision entries.
        :param distance: the distance that giskard is trying to keep from all objects
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        self.set_collision_entries([collision_entry])

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

    def plan_and_execute(self, wait=True):
        """
        :param wait: this function block if wait=True
        :type wait: bool
        :return: result from giskard
        :rtype: MoveResult
        """
        return self.send_goal(MoveGoal.PLAN_AND_EXECUTE, wait)

    def check_reachability(self, wait=True):
        """
        Not implemented
        :param wait: this function block if wait=True
        :type wait: bool
        :return: result from giskard
        :rtype: MoveResult
        """
        raise NotImplementedError('reachability check is not implemented')

    def plan(self, wait=True):
        """
        Plans, but doesn't execute the goal. Useful, if you just want to look at the planning ghost.
        :param wait: this function block if wait=True
        :type wait: bool
        :return: result from giskard
        :rtype: MoveResult
        """
        return self.send_goal(MoveGoal.PLAN_ONLY, wait)

    def send_goal(self, goal_type, wait=True):
        goal = self._get_goal()
        goal.type = goal_type
        if wait:
            self._client.send_goal_and_wait(goal)
            return self._client.get_result()
        else:
            self._client.send_goal(goal, feedback_cb=self._feedback_cb)

    def get_collision_entries(self):
        return self.cmd_seq

    def _get_goal(self):
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

    def get_result(self, timeout=rospy.Duration()):
        """
        Waits for giskardpy result and returns it. Only used when plan_and_execute was called with wait=False
        :type timeout: rospy.Duration
        :rtype: MoveResult
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

    def remove_object(self,
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
                timeout: float = 0) -> UpdateWorldResponse:
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

    def reattach_object(self,
                        name: str,
                        parent_link: str,
                        parent_link_group: str,
                        timeout: float = 0) -> UpdateWorldResponse:
        req = UpdateWorldRequest()
        req.operation = UpdateWorldRequest.REATTACH
        req.group_name = str(name)
        req.parent_link = parent_link
        req.parent_link_group = parent_link_group
        req.timeout = timeout
        return self._update_world_srv.call(req)

    def detach_object(self, object_name: str, timeout: float = 0):
        """
        Detach an object from the robot and add it back to the world.
        Careful though, you could amputate an arm be accident!
        :type object_name: str
        :return: UpdateWorldResponse
        """
        req = UpdateWorldRequest()
        req.timeout = timeout
        req.group_name = str(object_name)
        req.operation = req.DETACH
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

    def get_object_names(self):
        """
        returns the names of every object in the world
        :rtype: GetObjectNamesResponse
        """
        return self._get_object_names_srv()

    def get_object_info(self, name):
        """
        returns the joint state, joint state topic and pose of the object with the given name
        :type name: str
        :rtype: GetObjectInfoResponse
        """
        return self._get_object_info_srv(name)

    def get_controlled_joints(self, name='robot'):
        return self.get_object_info(name).controlled_joints

    def update_rviz_markers(self, object_names):
        """
        republishes visualization markers for rviz
        :type object_names: list
        :rtype: UpdateRvizMarkersResponse
        """
        return self._update_rviz_markers_srv(object_names)

    def get_attached_objects(self):
        """
        returns a list of all objects that are attached to the robot and the respective attachement points
        :rtype: GetAttachedObjectsResponse
        """
        return self._get_attached_objects_srv()
