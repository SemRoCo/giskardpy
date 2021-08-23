
import threading
from giskardpy.plugin import GiskardBehavior
from giskardpy.identifier import vs_goal
from py_trees import Status
import rospy
from giskardpy import tfwrapper

from geometry_msgs.msg import PoseStamped

class VisualServoingPlugin(GiskardBehavior):
    def __init__(self, name, goal_update_topic="/visual_servoing/goal_update"):
        super(VisualServoingPlugin, self).__init__(name)
        self.lock = threading.Lock()
        self.goal_update_topic = goal_update_topic
        self.goal_pose = None

    def setup(self, timeout=0.0):
        self.joint_state_sub = rospy.Subscriber(self.goal_update_topic, PoseStamped, self.cb, queue_size=1)
        return super(VisualServoingPlugin, self).setup(timeout)

    def cb(self, msg):
        self.lock.acquire()
        new_goal_pose = msg
        if not new_goal_pose.header.frame_id == "map":
            new_goal_pose = tfwrapper.transform_pose("map", new_goal_pose)
        rospy.loginfo("Updating goal pose")
        rospy.loginfo("From: {}".format(self.get_god_map().get_data(vs_goal)))
        rospy.loginfo("To: {}".format(new_goal_pose))
        self.goal_pose = new_goal_pose
        self.lock.release()

    @profile
    def update(self):
        #Todo: Get pose from clement

        self.lock.acquire()
        goal_pose = self.goal_pose
        self.goal_pose = None
        self.lock.release()

        if goal_pose is not None:
            self.get_god_map().set_data(vs_goal, goal_pose)

        return Status.RUNNING