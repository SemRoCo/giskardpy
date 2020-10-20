import json

import rospy
from actionlib import SimpleActionClient
from genpy import Message
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped, PointStamped
from giskard_msgs.msg import MoveAction, MoveGoal, WorldBody, CollisionEntry, MoveResult, Constraint, \
    MoveCmd, JointConstraint, CartesianConstraint
from giskard_msgs.srv import UpdateWorld, UpdateWorldRequest, UpdateWorldResponse, GetObjectInfo, GetObjectNames, \
    UpdateRvizMarkers, GetAttachedObjects, GetAttachedObjectsResponse, GetObjectNamesResponse
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import MarkerArray

from giskardpy.constraints import WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA
from giskardpy.urdf_object import URDFObject
from giskardpy.utils import position_dict_to_joint_states, make_world_body_box, make_world_body_cylinder
from rospy_message_converter.message_converter import convert_ros_message_to_dictionary


class GiskardWrapper(object):
    def __init__(self, giskard_topic=u'giskardpy/command', ns=u'giskardpy'):
        if giskard_topic is not None:
            self._client = SimpleActionClient(giskard_topic, MoveAction)
            self._update_world_srv = rospy.ServiceProxy(u'{}/update_world'.format(ns), UpdateWorld)
            self._get_object_names_srv = rospy.ServiceProxy(u'{}/get_object_names'.format(ns), GetObjectNames)
            self._get_object_info_srv = rospy.ServiceProxy(u'{}/get_object_info'.format(ns), GetObjectInfo)
            self._update_rviz_markers_srv = rospy.ServiceProxy(u'{}/update_rviz_markers'.format(ns), UpdateRvizMarkers)
            self._get_attached_objects_srv = rospy.ServiceProxy(u'{}/get_attached_objects'.format(ns), GetAttachedObjects)
            self._marker_pub = rospy.Publisher(u'visualization_marker_array', MarkerArray, queue_size=10)
            rospy.wait_for_service(u'{}/update_world'.format(ns))
            self._client.wait_for_server()
        self.robot_urdf = URDFObject(rospy.get_param(u'robot_description'))
        self.collisions = []
        self.clear_cmds()
        self._object_js_topics = {}
        rospy.sleep(.3)

    def get_robot_name(self):
        """
        :rtype: str
        """
        return self.robot_urdf.get_name()

    def get_root(self):
        """
        Returns the name of the robot's root link
        :rtype: str
        """
        return self.robot_urdf.get_root()

    def set_cart_goal(self, root_link, tip_link, goal_pose, max_linear_velocity=None, max_angular_velocity=None, weight=None):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose
        :param root_link: name of the root link of the kin chain
        :type root_link: str
        :param tip_link: name of the tip link of the kin chain
        :type tip_link: str
        :param goal: the goal pose
        :type goal: PoseStamped
        :param max_linear_velocity: m/s, default 0.1
        :type max_linear_velocity: float
        :param max_angular_velocity: rad/s, default 0.5
        :type max_angular_velocity: float
        :param weight: default WEIGHT_ABOVE_CA
        :type weight: float
        """
        self.set_translation_goal(root_link, tip_link, goal_pose, max_velocity=max_linear_velocity, weight=weight)
        self.set_rotation_goal(root_link, tip_link, goal_pose, max_velocity=max_angular_velocity, weight=weight)

    def set_translation_goal(self, root_link, tip_link, goal_pose, weight=None, max_velocity=None):
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
        if not max_velocity and not weight:
            constraint = CartesianConstraint()
            constraint.type = CartesianConstraint.TRANSLATION_3D
            constraint.root_link = str(root_link)
            constraint.tip_link = str(tip_link)
            constraint.goal = goal_pose
            self.cmd_seq[-1].cartesian_constraints.append(constraint)
        else:
            constraint = Constraint()
            constraint.type = u'CartesianPosition'
            params = {}
            params[u'root_link'] = root_link
            params[u'tip_link'] = tip_link
            params[u'goal'] = convert_ros_message_to_dictionary(goal_pose)
            if max_velocity:
                params[u'max_velocity'] = max_velocity
            if weight:
                params[u'weight'] = weight
            constraint.parameter_value_pair = json.dumps(params)
            self.cmd_seq[-1].constraints.append(constraint)

    def set_rotation_goal(self, root_link, tip_link, goal_pose, weight=None, max_velocity=None):
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
        if not max_velocity and not weight:
            constraint = CartesianConstraint()
            constraint.type = CartesianConstraint.ROTATION_3D
            constraint.root_link = str(root_link)
            constraint.tip_link = str(tip_link)
            constraint.goal = goal_pose
            self.cmd_seq[-1].cartesian_constraints.append(constraint)
        else:
            constraint = Constraint()
            constraint.type = u'CartesianOrientationSlerp'
            params = {}
            params[u'root_link'] = root_link
            params[u'tip_link'] = tip_link
            params[u'goal'] = convert_ros_message_to_dictionary(goal_pose)
            if max_velocity:
                params[u'max_velocity'] = max_velocity
            if weight:
                params[u'weight'] = weight
            constraint.parameter_value_pair = json.dumps(params)
            self.cmd_seq[-1].constraints.append(constraint)

    def set_joint_goal(self, goal_state, weight=None, max_velocity=None):
        """
        This goal will move the robots joint to the desired configuration.
        :param goal_state: Can either be a joint state messages or a dict mapping joint name to position. 
        :type goal_state: Union[JointState, dict]
        :param weight: default WEIGHT_BELOW_CA
        :type weight: float
        :param max_velocity: default is the default of the added joint goals
        :type max_velocity: float
        """
        if weight is None and max_velocity is None:
            constraint = JointConstraint()
            constraint.type = JointConstraint.JOINT
            if isinstance(goal_state, JointState):
                constraint.goal_state = goal_state
            else:
                for joint_name, joint_position in goal_state.items():
                    constraint.goal_state.name.append(joint_name)
                    constraint.goal_state.position.append(joint_position)
            self.cmd_seq[-1].joint_constraints.append(constraint)
        else:
            constraint = Constraint()
            constraint.type = JointConstraint.JOINT
            if isinstance(goal_state, JointState):
                goal_state = goal_state
            else:
                goal_state = JointState()
                for joint_name, joint_position in goal_state.items():
                    goal_state.name.append(joint_name)
                    goal_state.position.append(joint_position)
            params = {}
            params[u'goal_state'] = convert_ros_message_to_dictionary(goal_state)
            if weight is not None:
                params[u'weight'] = weight
            if max_velocity is not None:
                params[u'max_velocity'] = max_velocity
            constraint.parameter_value_pair = json.dumps(params)
            self.cmd_seq[-1].constraints.append(constraint)

    def align_planes(self, tip_link, tip_normal, root_link=None, root_normal=None, max_angular_velocity=None,
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
        :type weight: float
        """
        if root_link is None:
            root_link = self.get_root()
        if root_normal is None:
            root_normal = Vector3Stamped()
            root_normal.header.frame_id = self.get_root()
            root_normal.vector.z = 1

        params = {u'tip_link': tip_link,
                  u'tip_normal': tip_normal,
                  u'root_link': root_link,
                  u'root_normal': root_normal}
        if weight is not None:
            params[u'weight'] = weight
        if max_angular_velocity is not None:
            params[u'max_angular_velocity'] = max_angular_velocity
        self.set_json_goal(u'AlignPlanes', **params)

    def avoid_joint_limits(self, percentage=15, weight=WEIGHT_BELOW_CA):
        """
        This goal will push joints away from their position limits
        :param percentage: default 15, if limits are 0-100, the constraint will push into the 15-85 range
        :type percentage: float
        :param weight: default WEIGHT_BELOW_CA
        :type weight: float
        """
        self.set_json_goal(u'AvoidJointLimits', percentage=percentage, weight=weight)

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
        self.set_json_goal(u'CartesianVelocityLimit',
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
        self.set_json_goal(u'GraspBar',
                           root_link=root_link,
                           tip_link=tip_link,
                           tip_grasp_axis=tip_grasp_axis,
                           bar_center=bar_center,
                           bar_axis=bar_axis,
                           bar_length=bar_length,
                           max_linear_velocity=max_linear_velocity,
                           max_angular_velocity=max_angular_velocity,
                           weight=weight)

    def update_god_map(self, updates):
        """
        don't use, it's only for hacks :)
        """
        self.set_json_goal(u'UpdateGodMap', updates=updates)

    def pointing(self, tip_link, goal_point, root_link=None, pointing_axis=None, weight=None):
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
        kwargs = {u'tip_link': tip_link,
                  u'goal_point': goal_point}
        if root_link is not None:
            kwargs[u'root_link'] = root_link
        if pointing_axis is not None:
            kwargs[u'pointing_axis'] = pointing_axis
        if weight is not None:
            kwargs[u'weight'] = weight
        kwargs[u'goal_point'] = goal_point
        self.set_json_goal(u'Pointing', **kwargs)

    def set_json_goal(self, constraint_type, **kwargs):
        """
        Set a goal for any of the goals defined in Constraints.py
        :param constraint_type: Name of the Goal
        :type constraint_type: str
        :param kwargs: maps constraint parameter names to values. Values should be float, str or ros messages.
        :type kwargs: dict
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

    def allow_collision(self, robot_links=(CollisionEntry.ALL,), body_b=CollisionEntry.ALL,
                        link_bs=(CollisionEntry.ALL,)):
        """
        :param robot_links: list of robot link names as str, None or empty list means all
        :type robot_links: list
        :param body_b: name of the other body, use the robots name to modify self collision behavior, empty string means all bodies
        :type body_b: str
        :param link_bs: list of link name of body_b, None or empty list means all
        :type link_bs: list
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.robot_links = [str(x) for x in robot_links]
        collision_entry.body_b = str(body_b)
        collision_entry.link_bs = [str(x) for x in link_bs]
        self.set_collision_entries([collision_entry])

    def avoid_collision(self, min_dist, robot_links=(CollisionEntry.ALL,), body_b=CollisionEntry.ALL,
                        link_bs=(CollisionEntry.ALL,)):
        """
        :param min_dist: the distance giskard is trying to keep between specified links
        :type min_dist: float
        :param robot_links: list of robot link names as str, None or empty list means all
        :type robot_links: list
        :param body_b: name of the other body, use the robots name to modify self collision behavior, empty string means all bodies
        :type body_b: str
        :param link_bs: list of link name of body_b, None or empty list means all
        :type link_bs: list
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = min_dist
        collision_entry.robot_links = [str(x) for x in robot_links]
        collision_entry.body_b = str(body_b)
        collision_entry.link_bs = [str(x) for x in link_bs]
        self.set_collision_entries([collision_entry])

    def allow_all_collisions(self):
        """
        Allows all collisions for next goal.
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.robot_links = [CollisionEntry.ALL]
        collision_entry.body_b = CollisionEntry.ALL
        collision_entry.link_bs = [CollisionEntry.ALL]
        self.set_collision_entries([collision_entry])

    def allow_self_collision(self):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.robot_links = [CollisionEntry.ALL]
        collision_entry.body_b = self.get_robot_name()
        collision_entry.link_bs = [CollisionEntry.ALL]
        self.set_collision_entries([collision_entry])

    def avoid_self_collision(self):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.robot_links = [CollisionEntry.ALL]
        collision_entry.body_b = self.get_robot_name()
        collision_entry.link_bs = [CollisionEntry.ALL]
        self.set_collision_entries([collision_entry])

    def avoid_all_collisions(self, distance=0.05):
        """
        Avoids all collisions for next goal. The distance will override anything from the config file.
        If you don't want to override the distance, don't call this function. Avoid all is the default, if you don't
        add any collision entries.
        :param distance: the distance that giskard is trying to keep from all objects
        :type distance: float
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.robot_links = [CollisionEntry.ALL]
        collision_entry.body_b = CollisionEntry.ALL
        collision_entry.link_bs = [CollisionEntry.ALL]
        collision_entry.min_dist = distance
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

    def plan_and_execute(self, wait=True):
        """
        :param wait: this function block if wait=True
        :type wait: bool
        :return: result from giskard
        :rtype: MoveResult
        """
        goal = self._get_goal()
        if wait:
            self._client.send_goal_and_wait(goal)
            return self._client.get_result()
        else:
            self._client.send_goal(goal)

    def check_reachability(self, wait=True):
        """
        Not implemented
        :param wait: this function block if wait=True
        :type wait: bool
        :return: result from giskard
        :rtype: MoveResult
        """
        goal = self._get_goal()
        goal.type = MoveGoal.CHECK_REACHABILITY
        if wait:
            self._client.send_goal_and_wait(goal)
            return self._client.get_result()
        else:
            self._client.send_goal(goal)

    def plan(self, wait=True):
        """
        Plans, but doesn't execute the goal. Useful, if you just want to look at the planning ghost.
        :param wait: this function block if wait=True
        :type wait: bool
        :return: result from giskard
        :rtype: MoveResult
        """
        goal = self._get_goal()
        goal.type = MoveGoal.PLAN_ONLY
        if wait:
            self._client.send_goal_and_wait(goal)
            return self._client.get_result()
        else:
            self._client.send_goal(goal)

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
        self._client.wait_for_result(timeout)
        return self._client.get_result()

    def clear_world(self):
        """
        Removes any objects and attached objects from Giskard's world and reverts the robots urdf to what it got from
        the parameter server.
        :rtype: UpdateWorldResponse
        """
        req = UpdateWorldRequest(UpdateWorldRequest.REMOVE_ALL, WorldBody(), False, PoseStamped())
        return self._update_world_srv.call(req)

    def remove_object(self, name):
        """
        :param name:
        :type name: str
        :return:
        :rtype: UpdateWorldResponse
        """
        object = WorldBody()
        object.name = str(name)
        req = UpdateWorldRequest(UpdateWorldRequest.REMOVE, object, False, PoseStamped())
        return self._update_world_srv.call(req)

    def add_box(self, name=u'box', size=(1, 1, 1), frame_id=u'map', position=(0, 0, 0), orientation=(0, 0, 0, 1),
                pose=None):
        """
        If pose is used, frame_id, position and orientation are ignored.
        :type name: str
        :param size: (x length, y length, z length) in m
        :type size: list
        :type frame_id: str
        :type position: list
        :type orientation: list
        :type pose: PoseStamped
        :rtype: UpdateWorldResponse
        """
        box = make_world_body_box(name, size[0], size[1], size[2])
        if pose is None:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = str(frame_id)
            pose.pose.position = Point(*position)
            pose.pose.orientation = Quaternion(*orientation)
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, box, False, pose)
        return self._update_world_srv.call(req)

    def add_sphere(self, name=u'sphere', size=1, frame_id=u'map', position=(0, 0, 0), orientation=(0, 0, 0, 1),
                   pose=None):
        """
        If pose is used, frame_id, position and orientation are ignored.
        :type name: str
        :param size: radius in m
        :type size: list
        :type frame_id: str
        :type position: list
        :type orientation: list
        :type pose: PoseStamped
        :rtype: UpdateWorldResponse
        """
        object = WorldBody()
        object.type = WorldBody.PRIMITIVE_BODY
        object.name = str(name)
        if pose is None:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = str(frame_id)
            pose.pose.position = Point(*position)
            pose.pose.orientation = Quaternion(*orientation)
        object.shape.type = SolidPrimitive.SPHERE
        object.shape.dimensions.append(size)
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, object, False, pose)
        return self._update_world_srv.call(req)

    def add_mesh(self, name=u'mesh', mesh=u'', frame_id=u'map', position=(0, 0, 0), orientation=(0, 0, 0, 1),
                 pose=None):
        """
        If pose is used, frame_id, position and orientation are ignored.
        :type name: str
        :param mesh: path to the meshes location. e.g. package://giskardpy/test/urdfs/meshes/bowl_21.obj
        :type frame_id: str
        :type position: list
        :type orientation: list
        :type pose: PoseStamped
        :rtype: UpdateWorldResponse
        """
        object = WorldBody()
        object.type = WorldBody.MESH_BODY
        object.name = str(name)
        if pose is None:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = str(frame_id)
            pose.pose.position = Point(*position)
            pose.pose.orientation = Quaternion(*orientation)
        object.mesh = mesh
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, object, False, pose)
        return self._update_world_srv.call(req)

    def add_cylinder(self, name=u'cylinder', height=1, radius=1, frame_id=u'map', position=(0, 0, 0), orientation=(0, 0, 0, 1),
                     pose=None):
        """
        If pose is used, frame_id, position and orientation are ignored.
        :type name: str
        :param height: in m
        :type height: float
        :param radius: in m
        :type radius: float
        :type frame_id: str
        :type position: list
        :type orientation: list
        :type pose: PoseStamped
        :rtype: UpdateWorldResponse
        """
        object = WorldBody()
        object.type = WorldBody.PRIMITIVE_BODY
        object.name = str(name)
        if pose is None:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = str(frame_id)
            pose.pose.position = Point(*position)
            pose.pose.orientation = Quaternion(*orientation)
        object.shape.type = SolidPrimitive.CYLINDER
        object.shape.dimensions = [0,0]
        object.shape.dimensions[SolidPrimitive.CYLINDER_HEIGHT] = height
        object.shape.dimensions[SolidPrimitive.CYLINDER_RADIUS] = radius
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, object, False, pose)
        return self._update_world_srv.call(req)

    def attach_box(self, name=u'box', size=None, frame_id=None, position=None, orientation=None, pose=None):
        """
        Add a box to the world and attach it to the robot at frame_id.
        If pose is used, frame_id, position and orientation are ignored.
        :type name: str
        :type size: list
        :type frame_id: str
        :type position: list
        :type orientation: list
        :rtype: UpdateWorldResponse
        """

        box = make_world_body_box(name, size[0], size[1], size[2])
        if pose is None:
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = str(frame_id) if frame_id is not None else u'map'
            pose.pose.position = Point(*(position if position is not None else [0, 0, 0]))
            pose.pose.orientation = Quaternion(*(orientation if orientation is not None else [0, 0, 0, 1]))

        req = UpdateWorldRequest(UpdateWorldRequest.ADD, box, True, pose)
        return self._update_world_srv.call(req)

    def attach_cylinder(self, name=u'cylinder', height=1, radius=1, frame_id=None, position=None, orientation=None):
        """
        Add a cylinder to the world and attach it to the robot at frame_id.
        If pose is used, frame_id, position and orientation are ignored.
        :type name: str
        :type size: list
        :type frame_id: str
        :type position: list
        :type orientation: list
        :rtype: UpdateWorldResponse
        """
        cylinder = make_world_body_cylinder(name, height, radius)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(frame_id) if frame_id is not None else u'map'
        pose.pose.position = Point(*(position if position is not None else [0, 0, 0]))
        pose.pose.orientation = Quaternion(*(orientation if orientation is not None else [0, 0, 0, 1]))

        req = UpdateWorldRequest(UpdateWorldRequest.ADD, cylinder, True, pose)
        return self._update_world_srv.call(req)

    def attach_object(self, name, link_frame_id):
        """
        Attach an already existing object at link_frame_id of the robot.
        :type name: str
        :param link_frame_id: name of a robot link
        :type link_frame_id: str
        :return: UpdateWorldResponse
        """
        req = UpdateWorldRequest()
        req.rigidly_attached = True
        req.body.name = name
        req.pose.header.frame_id = link_frame_id
        req.operation = UpdateWorldRequest.ADD
        return self._update_world_srv.call(req)

    def detach_object(self, object_name):
        """
        Detach an object from the robot and add it back to the world.
        Careful though, you could amputate an arm be accident!
        :type object_name: str
        :return: UpdateWorldResponse
        """
        req = UpdateWorldRequest()
        req.body.name = object_name
        req.operation = req.DETACH
        return self._update_world_srv.call(req)

    def add_urdf(self, name, urdf, pose, js_topic=u'', set_js_topic=None):
        """
        Adds a urdf to the world
        :param name: name it will have in the world
        :type name: str
        :param urdf: urdf as string, no path
        :type urdf: str
        :type pose: PoseStamped
        :param js_topic: Giskard will listen on that topic for joint states and update the urdf accordingly
        :type js_topic: str
        :param set_js_topic: A topic that the python wrapper will use to set the urdf joint state.
                                If None, set_js_topic == js_topic
        :type set_js_topic: str
        :return: UpdateWorldResponse
        """
        if set_js_topic is None:
            set_js_topic = js_topic
        urdf_body = WorldBody()
        urdf_body.name = str(name)
        urdf_body.type = WorldBody.URDF_BODY
        urdf_body.urdf = str(urdf)
        urdf_body.joint_state_topic = str(js_topic)
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, urdf_body, False, pose)
        if js_topic:
            # FIXME publisher has to be removed, when object gets deleted
            # FIXME there could be sync error, if objects get added/removed by something else
            self._object_js_topics[name] = rospy.Publisher(set_js_topic, JointState, queue_size=10)
        return self._update_world_srv.call(req)

    def set_object_joint_state(self, object_name, joint_states):
        """
        :type object_name: str
        :param joint_states: joint state message or a dict that maps joint name to position
        :type joint_states: Union[JointState, dict]
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

    def update_rviz_markers(self, object_names):
        """
        republishes visualization markers for rviz
        :type name: list
        :rtype: UpdateRvizMarkersResponse
        """
        return self._update_rviz_markers_srv(object_names)

    def get_attached_objects(self):
        """
        returns a list of all objects that are attached to the robot and the respective attachement points
        :rtype: GetAttachedObjectsResponse
        """
        return self._get_attached_objects_srv()
