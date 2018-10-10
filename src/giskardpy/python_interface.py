import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveAction, MoveCmd, Controller, MoveGoal, WorldBody, CollisionEntry
from giskard_msgs.srv import UpdateWorld, UpdateWorldRequest, UpdateWorldResponse
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import MarkerArray
from giskardpy.utils import dict_to_joint_states


class GiskardWrapper(object):
    def __init__(self, giskard_topic=u'giskardpy/command', ns=u'giskard', joint_gain=10, joint_max_speed=1):
        if giskard_topic is not None:
            self.client = SimpleActionClient(giskard_topic, MoveAction)
            self.update_world = rospy.ServiceProxy(u'{}/update_world'.format(ns), UpdateWorld)
            self.marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
            rospy.wait_for_service(u'{}/update_world'.format(ns))
            self.client.wait_for_server()
        self.tip_to_root = {}
        self.robot_name = rospy.get_param(u'robot_description').split('\n',1)[1].split('" ',1)[0].split('"')[1]
        self.collisions = []
        self.clear_cmds()
        self.object_js_topics = {}
        self.joint_gain = joint_gain
        self.joint_max_speed = joint_max_speed
        rospy.sleep(.3)

    def set_cart_goal(self, root, tip, pose_stamped):
        """
        :param tip:
        :type tip: str
        :param pose_stamped:
        :type pose_stamped: PoseStamped
        """
        self.set_tranlation_goal(root, tip, pose_stamped)
        self.set_rotation_goal(root, tip, pose_stamped)


    def set_tranlation_goal(self, root, tip, pose_stamped, p_gain=3, max_speed=0.1):
        """
        :param tip:
        :type tip: str
        :param pose_stamped:
        :type pose_stamped: PoseStamped
        """
        controller = Controller()
        controller.root_link = str(root)
        controller.tip_link = str(tip)
        controller.goal_pose = pose_stamped
        controller.type = Controller.TRANSLATION_3D
        controller.weight = 1
        controller.max_speed = max_speed
        controller.p_gain = p_gain
        self.cmd_seq[-1].controllers.append(controller)

    def set_rotation_goal(self, root, tip, pose_stamped, p_gain=3, max_speed=1.0):
        """
        :param tip:
        :type tip: str
        :param pose_stamped:
        :type pose_stamped: PoseStamped
        """
        controller = Controller()
        controller.root_link = str(root)
        controller.tip_link = str(tip)
        controller.goal_pose = pose_stamped
        controller.type = Controller.ROTATION_3D
        controller.weight = 1
        controller.max_speed = max_speed
        controller.p_gain = p_gain
        self.cmd_seq[-1].controllers.append(controller)

    def set_joint_goal(self, joint_state):
        """
        :param joint_state:
        :type joint_state: dict
        """
        controller = Controller()
        controller.type = Controller.JOINT
        controller.weight = 1
        controller.p_gain = self.joint_gain
        controller.max_speed = self.joint_max_speed
        if isinstance(joint_state, dict):
            for joint_name, joint_position in joint_state.items():
                controller.goal_state.name.append(joint_name)
                controller.goal_state.position.append(joint_position)
        elif isinstance(joint_state, JointState):
            controller.goal_state = joint_state
        self.cmd_seq[-1].controllers.append(controller)

    def set_collision_entries(self, collisions):
        self.cmd_seq[-1].collisions.extend(collisions)

    def avoid_collision(self, min_dist, robot_links=None, body_b=u'', link_bs=None):
        if robot_links is None:
            robot_links = []
        if link_bs is None:
            link_bs = []
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = min_dist
        collision_entry.robot_links = [str(x) for x in robot_links]
        collision_entry.body_b = str(body_b)
        collision_entry.link_bs = [str(x) for x in link_bs]
        self.set_collision_entries([collision_entry])

    def allow_all_collisions(self):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_ALL_COLLISIONS
        self.set_collision_entries([collision_entry])

    def add_cmd(self, max_trajectory_length=20):
        move_cmd = MoveCmd()
        # move_cmd.max_trajectory_length = max_trajectory_length
        self.cmd_seq.append(move_cmd)

    def clear_cmds(self):
        self.cmd_seq = []
        self.add_cmd()

    def plan_and_execute(self, wait=True):
        """
        :return:
        :rtype: giskard_msgs.msg._MoveResult.MoveResult
        """
        goal = self._get_goal()
        if wait:
            self.client.send_goal_and_wait(goal)
            return self.client.get_result()
        else:
            self.client.send_goal(goal)

    def get_collision_entries(self):
        return self.cmd_seq

    def _get_goal(self):
        goal = MoveGoal()
        goal.cmd_seq = self.cmd_seq
        goal.type = MoveGoal.PLAN_AND_EXECUTE
        self.clear_cmds()
        return goal

    def interrupt(self):
        self.client.cancel_goal()

    def get_result(self,  timeout=rospy.Duration()):
        self.client.wait_for_result(timeout)
        return self.client.get_result()

    def clear_world(self):
        """
        :rtype: UpdateWorldResponse
        """
        req = UpdateWorldRequest(UpdateWorldRequest.REMOVE_ALL, WorldBody(), False, PoseStamped())
        return self.update_world.call(req)

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
        return self.update_world.call(req)

    def make_box(self, name=u'box', size=(1,1,1)):
        box = WorldBody()
        box.type = WorldBody.PRIMITIVE_BODY
        box.name = str(name)
        box.shape.type = SolidPrimitive.BOX
        box.shape.dimensions.append(size[0])
        box.shape.dimensions.append(size[1])
        box.shape.dimensions.append(size[2])
        return box

    def add_box(self, name=u'box', size=(1, 1, 1), frame_id=u'map', position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        box = self.make_box(name, size)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(frame_id)
        pose.pose.position = Point(*position)
        pose.pose.orientation = Quaternion(*orientation)
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, box, False, pose)
        return self.update_world.call(req)

    def add_sphere(self, name=u'sphere', size=1, frame_id=u'map', position=(0,0,0), orientation=(0,0,0,1)):
        object = WorldBody()
        object.type = WorldBody.PRIMITIVE_BODY
        object.name = str(name)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(frame_id)
        pose.pose.position = Point(*position)
        pose.pose.orientation = Quaternion(*orientation)
        object.shape.type = SolidPrimitive.SPHERE
        object.shape.dimensions.append(size)
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, object, False, pose)
        return self.update_world.call(req)

    def add_cylinder(self, name=u'cylinder', size=(1,1), frame_id=u'map', position=(0,0,0), orientation=(0,0,0,1)):
        object = WorldBody()
        object.type = WorldBody.PRIMITIVE_BODY
        object.name = str(name)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(frame_id)
        pose.pose.position = Point(*position)
        pose.pose.orientation = Quaternion(*orientation)
        object.shape.type = SolidPrimitive.CYLINDER
        object.shape.dimensions.append(size[0])
        object.shape.dimensions.append(size[1])
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, object, False, pose)
        return self.update_world.call(req)

    def attach_box(self, name=u'box', size=(1, 1, 1), frame_id=u'map', position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        """
        :param name:
        :param size:
        :param frame_id:
        :param position:
        :param orientation:
        :rtype: UpdateWorldResponse
        """
        box = self.make_box(name, size)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = str(frame_id)
        pose.pose.position = Point(*position)
        pose.pose.orientation = Quaternion(*orientation)

        req = UpdateWorldRequest(UpdateWorldRequest.ADD, box, True, pose)
        return self.update_world.call(req)

    def add_urdf(self, name, urdf, js_topic, pose):
        urdf_body = WorldBody()
        urdf_body.name = str(name)
        urdf_body.type = WorldBody.URDF_BODY
        urdf_body.urdf = str(urdf)
        urdf_body.joint_state_topic = str(js_topic)
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, urdf_body, False, pose)
        self.object_js_topics[name] = rospy.Publisher(js_topic, JointState, queue_size=10)
        return self.update_world.call(req)

    def set_object_joint_state(self, object_name, joint_states):
        if isinstance(joint_states, dict):
            joint_states = dict_to_joint_states(joint_states)
        self.object_js_topics[object_name].publish(joint_states)

    def disable_self_collision(self):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.min_dist = 1
        collision_entry.body_b = self.robot_name
        self.set_collision_entries([collision_entry])