from actionlib import SimpleActionClient
from geometry_msgs.msg import PoseStamped
from giskard_msgs.msg import MoveAction, MoveCmd, Controller, MoveGoal


class GiskardWrapper(object):
    def __init__(self, root_tips):
        self.client = SimpleActionClient('qp_controller/command', MoveAction)
        self.client.wait_for_server()
        self.tip_to_root = {}
        self.controller = []
        self.collisions = []
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
        self.controller.append(controller)
        controller = Controller()
        controller.root_link = self.tip_to_root[tip]
        controller.tip_link = tip
        controller.goal_pose = pose_stamped
        controller.type = Controller.ROTATION_3D
        controller.weight = 1
        controller.threshold_value = 0.5
        controller.p_gain = 3
        self.controller.append(controller)

    def set_collision_entries(self, collisions):
        self.collisions = collisions

    def send_goals(self):
        cmd = MoveCmd()
        cmd.controllers = self.controller
        cmd.collisions = self.collisions
        goal = MoveGoal()
        goal.cmd_seq.append(cmd)
        goal.type = MoveGoal.PLAN_AND_EXECUTE
        self.controller = []
        self.collisions = []
        return self.client.send_goal_and_wait(goal)
