import rospy
from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from giskard_msgs.msg import MoveAction, MoveCmd, Controller, MoveGoal, WorldBody
from giskard_msgs.srv import UpdateWorld, UpdateWorldRequest
from shape_msgs.msg import SolidPrimitive


class GiskardWrapper(object):
    def __init__(self, root_tips):
        self.client = SimpleActionClient('qp_controller/command', MoveAction)
        self.update_world = rospy.ServiceProxy('muh/update_world', UpdateWorld)
        rospy.wait_for_service('muh/update_world')
        self.client.wait_for_server()
        self.tip_to_root = {}
        self.collisions = []
        self.clear_cmds()
        for root, tip in root_tips:
            self.tip_to_root[tip] = root

    def set_cart_goal(self, tip, pose_stamped):
        """
        :param tip:
        :type tip: str
        :param pose_stamped:
        :type pose_stamped: PoseStamped
        """
        controller = Controller()
        controller.root_link = self.tip_to_root[tip]
        controller.tip_link = tip
        controller.goal_pose = pose_stamped
        controller.type = Controller.TRANSLATION_3D
        controller.weight = 1
        controller.threshold_value = 0.3
        controller.p_gain = 3
        self.cmd_seq[-1].controllers.append(controller)
        controller = Controller()
        controller.root_link = self.tip_to_root[tip]
        controller.tip_link = tip
        controller.goal_pose = pose_stamped
        controller.type = Controller.ROTATION_3D
        controller.weight = 1
        controller.threshold_value = 0.5
        controller.p_gain = 3
        self.cmd_seq[-1].controllers.append(controller)

    def set_collision_entries(self, collisions):
        self.cmd_seq[-1].collisions.extend(collisions)

    def add_cmd(self):
        self.cmd_seq.append(MoveCmd())

    def clear_cmds(self):
        self.cmd_seq = []
        self.add_cmd()

    def plan_and_execute(self):
        """
        :return:
        :rtype: giskard_msgs.msg._MoveResult.MoveResult
        """
        goal = MoveGoal()
        goal.cmd_seq = self.cmd_seq
        goal.type = MoveGoal.PLAN_AND_EXECUTE
        self.clear_cmds()
        self.client.send_goal_and_wait(goal)
        return self.client.get_result()

    def clear_world(self):
        req = UpdateWorldRequest(UpdateWorldRequest.REMOVE_ALL, WorldBody(), False, PoseStamped())
        self.update_world.call(req)

    def add_box(self, name='box', size=(1,1,1), position=(0,0,0), orientation=(0,0,0,1)):
        box = WorldBody()
        box.type = WorldBody.PRIMITIVE_BODY
        box.name = name
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = 'map'
        pose.pose.position = Point(*position)
        pose.pose.orientation = Quaternion(*orientation)
        box.shape.type = SolidPrimitive.BOX
        box.shape.dimensions.append(size[0])
        box.shape.dimensions.append(size[1])
        box.shape.dimensions.append(size[2])
        req = UpdateWorldRequest(UpdateWorldRequest.ADD, box, False, pose)
        self.update_world.call(req)