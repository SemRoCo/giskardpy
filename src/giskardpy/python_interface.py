import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveAction, MoveCmd, Controller, MoveGoal, WorldBody, CollisionEntry
from giskard_msgs.srv import UpdateWorld, UpdateWorldRequest
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray

from giskardpy.object import to_marker, from_msg
from giskardpy.tfwrapper import lookup_transform


class GiskardWrapper(object):
    def __init__(self, root_tips):
        self.client = SimpleActionClient('qp_controller/command', MoveAction)
        self.update_world = rospy.ServiceProxy('giskard/update_world', UpdateWorld)
        self.marker_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        rospy.wait_for_service('giskard/update_world')
        self.client.wait_for_server()
        self.tip_to_root = {}
        self.collisions = []
        self.clear_cmds()
        for root, tip in root_tips:
            self.tip_to_root[tip] = root
        rospy.sleep(.3)

    def set_cart_goal(self, tip, pose_stamped, root=None):
        """
        :param tip:
        :type tip: str
        :param pose_stamped:
        :type pose_stamped: PoseStamped
        """
        self.set_tranlation_goal(tip, pose_stamped, root)
        self.set_rotation_goal(tip, pose_stamped, root)


    def set_tranlation_goal(self, tip, pose_stamped, root=None):
        """
        :param tip:
        :type tip: str
        :param pose_stamped:
        :type pose_stamped: PoseStamped
        """
        controller = Controller()
        controller.root_link = self.tip_to_root[tip] if root is None else root
        controller.tip_link = tip
        controller.goal_pose = pose_stamped
        controller.type = Controller.TRANSLATION_3D
        controller.weight = 1
        controller.max_speed = 0.3
        controller.p_gain = 3
        self.cmd_seq[-1].controllers.append(controller)

    def set_rotation_goal(self, tip, pose_stamped, root=None):
        """
        :param tip:
        :type tip: str
        :param pose_stamped:
        :type pose_stamped: PoseStamped
        """
        controller = Controller()
        controller.root_link = self.tip_to_root[tip] if root is None else root
        controller.tip_link = tip
        controller.goal_pose = pose_stamped
        controller.type = Controller.ROTATION_3D
        controller.weight = 1
        controller.max_speed = 1.0
        controller.p_gain = 3
        self.cmd_seq[-1].controllers.append(controller)


    def set_joint_goal(self, joint_state):
        """
        :param joint_state:
        :type joint_state: dict
        """
        controller = Controller()
        controller.type = Controller.JOINT
        controller.weight = 1
        controller.p_gain = 10
        controller.max_speed = 1
        if isinstance(joint_state, dict):
            for joint_name, joint_position in joint_state.items():
                controller.goal_state.name.append(joint_name)
                controller.goal_state.position.append(joint_position)
        elif isinstance(joint_state, JointState):
            controller.goal_state = joint_state
        self.cmd_seq[-1].controllers.append(controller)

    def set_collision_entries(self, collisions):
        self.cmd_seq[-1].collisions.extend(collisions)

    def avoid_collision(self, min_dist, robot_link='', body_b='', link_b=''):
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.min_dist = min_dist
        collision_entry.robot_link = robot_link
        collision_entry.body_b = body_b
        collision_entry.link_b = link_b
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
        goal = MoveGoal()
        goal.cmd_seq = self.cmd_seq
        goal.type = MoveGoal.PLAN_AND_EXECUTE
        self.clear_cmds()
        if wait:
            self.client.send_goal_and_wait(goal)
            return self.client.get_result()
        else:
            self.client.send_goal(goal)

    def interrupt(self):
        self.client.cancel_goal()

    def get_result(self,  timeout=rospy.Duration()):
        self.client.wait_for_result(timeout)
        return self.client.get_result()

    def clear_world(self):
        req = UpdateWorldRequest(UpdateWorldRequest.REMOVE_ALL, WorldBody(), False, PoseStamped())
        self.update_world.call(req)

    def remove_object(self, name):
        """
        :param name:
        :type name: str
        :return:
        :rtype: giskard_msgs.srv._UpdateWorld.UpdateWorldResponse
        """
        object = WorldBody()
        object.name = name
        req = UpdateWorldRequest(UpdateWorldRequest.REMOVE, object, False, PoseStamped())
        return self.update_world.call(req)

    def add_box(self, name='box', size=(1,1,1), frame_id='map', position=(0,0,0), orientation=(0,0,0,1)):
        box = WorldBody()
        box.type = WorldBody.PRIMITIVE_BODY
        box.name = name
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame_id
        pose.pose.position = Point(*position)
        pose.pose.orientation = Quaternion(*orientation)
        box.shape.type = SolidPrimitive.BOX
        box.shape.dimensions.append(size[0])
        box.shape.dimensions.append(size[1])
        box.shape.dimensions.append(size[2])
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, box, False, pose)
        world_object = from_msg(box)
        ma = to_marker(world_object)
        ma.markers[0].pose = pose.pose

        self.marker_pub.publish(ma)
        return self.update_world.call(req)

    def add_sphere(self, name='sphere', size=1, frame_id='map', position=(0,0,0), orientation=(0,0,0,1)):
        object = WorldBody()
        object.type = WorldBody.PRIMITIVE_BODY
        object.name = name
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame_id
        pose.pose.position = Point(*position)
        pose.pose.orientation = Quaternion(*orientation)
        object.shape.type = SolidPrimitive.SPHERE
        object.shape.dimensions.append(size)
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, object, False, pose)
        return self.update_world.call(req)

    def add_cylinder(self, name='cylinder', size=(1,1), frame_id='map', position=(0,0,0), orientation=(0,0,0,1)):
        object = WorldBody()
        object.type = WorldBody.PRIMITIVE_BODY
        object.name = name
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame_id
        pose.pose.position = Point(*position)
        pose.pose.orientation = Quaternion(*orientation)
        object.shape.type = SolidPrimitive.CYLINDER
        object.shape.dimensions.append(size[0])
        object.shape.dimensions.append(size[1])
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, object, False, pose)
        return self.update_world.call(req)

    def add_cone(self, name='cone', size=(1,1), frame_id='map', position=(0,0,0), orientation=(0,0,0,1)):
        object = WorldBody()
        object.type = WorldBody.PRIMITIVE_BODY
        object.name = name
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame_id
        pose.pose.position = Point(*position)
        pose.pose.orientation = Quaternion(*orientation)
        object.shape.type = SolidPrimitive.CONE
        object.shape.dimensions.append(size[0])
        object.shape.dimensions.append(size[1])
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, object, False, pose)
        return self.update_world.call(req)

    def attach_object(self, name):
        # TODO implement me
        raise NotImplementedError


    def add_urdf(self, name, urdf, js_topic, map_frame, root_frame):
        urdf_body = WorldBody()
        urdf_body.name = name
        urdf_body.type = WorldBody.URDF_BODY
        urdf_body.urdf = urdf
        urdf_body.joint_state_topic = js_topic
        transform = lookup_transform(map_frame, root_frame)
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, urdf_body, False, transform)
        return self.update_world.call(req)
