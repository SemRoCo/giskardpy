from enum import Enum
from typing import Dict

from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortGoal

from fancylog import Logger
import geometry_msgs.msg
import numpy as np
import robokudo_msgs.msg
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped, Vector3Stamped
from tmc_msgs.msg import TalkRequestActionGoal

from giskardpy.python_interface import GiskardWrapper
import actionlib
from robokudo_msgs.msg import QueryAction, QueryGoal, QueryResult, ShapeSize
import giskardpy.utils.tfwrapper as tf


class Gripper:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('/hsrb/gripper_controller/grasp',
                                                   GripperApplyEffortAction)
        self.client.wait_for_server()

    def send_goal(self, effort: float):
        goal = GripperApplyEffortGoal()
        goal.effort = effort
        self.client.send_goal(goal)
        self.client.wait_for_server()

    def close(self):
        self.send_goal(-0.5)

    def open(self):
        self.send_goal(0.5)


class ObjectEntry:
    def __init__(self, pose: geometry_msgs.msg.PoseStamped, classname: str, size: np.ndarray, object_id: str) -> None:
        """
        Initialize an object entry.

        Parameters
        ----------
        pose : numpy.ndarray
            3D numpy array representing the translation vector of the object.
        classname : str
            The class name of the object.
        size : numpy.ndarray
            3D numpy array representing the size of the object.
        object_id : str
            The UUID of the object.
        """
        self.pose_in_cam: geometry_msgs.msg.PoseStamped = pose
        self.pose: geometry_msgs.msg.PoseStamped = tf.transform_pose('map', pose)
        self.classname: str = classname
        self.size: np.ndarray = size
        self.uuid: str = object_id

    @property
    def map_P_object(self) -> np.ndarray:
        return np.array([self.pose.pose.position.x,
                         self.pose.pose.position.y,
                         self.pose.pose.position.z])

class PerceptionTask(Enum):
    HUMAN_POINTING = 1
    OBJECT_DETECTION3D = 2
    OBJECT_DETECTION2D = 3
    HUMAN_TRACKING = 4
    FREE_SPACE = 5


# def find_leftmost_object():
#     # Check in the Belief state which object is the leftmost in cam coordinates
#     min_key = min(objbelief_state, key=lambda k: objbelief_state[k].pose_in_cam.pose.position[1])

# def find_rightmost_object():
#     # Check in the Belief state which object is the leftmost in cam coordinates
#     max_key = max(objbelief_state, key=lambda k: objbelief_state[k].pose_in_cam.pose.position[1])


default_pose = {
    'arm_flex_joint': 0.0,
    'arm_lift_joint': 0.0,
    'arm_roll_joint': 0.0,
    'head_pan_joint': 0.0,
    'head_tilt_joint': 0.0,
    'wrist_flex_joint': 0.0,
    'wrist_roll_joint': 0.0,
}

detect_pose = {
    'arm_flex_joint': -2.015608106413679,
    'arm_lift_joint': 0.14796888076205927,
    'arm_roll_joint': -1.5708215665884175,
    'head_pan_joint': 6.134348425668179e-05,
    'head_tilt_joint': -0.5432743806702237,
    'wrist_flex_joint': -1.5649939187217963,
    'wrist_roll_joint': -0.0769292576107441,
}

carry_pose = {
    'head_pan_joint': 0.0,
    'head_tilt_joint': -0.65,
    'arm_lift_joint': 0.0,
    'arm_flex_joint': -0.43,
    'arm_roll_joint': 0.0,
    'wrist_flex_joint': -1.17,
    'wrist_roll_joint': 0,
}

def hsr_say(pub, message:str=""):
    goal_msg = TalkRequestActionGoal()
    goal_msg.header.stamp = rospy.Time.now()
    goal_msg.goal.data.language = 1
    goal_msg.goal.data.sentence = message
    pub.publish(goal_msg)


class QueryActionClient:
    def __init__(self):
        # Create an action client connected to the "Query" action server
        self.client = actionlib.SimpleActionClient('robokudo/query', QueryAction)

        # Wait for the action server to start
        self.client.wait_for_server()

    def send_goal(self, perception_task: PerceptionTask, location: str = ""):
        """
        Send a goal to the action server.
        :param perception_task: Keyword of the perception task
        """
        goal = QueryGoal()

        if perception_task == PerceptionTask.HUMAN_POINTING:
            goal.type = "detect"
            goal.obj.type = "human"
            goal.obj.attribute.append("pointing")
        elif perception_task == PerceptionTask.OBJECT_DETECTION3D:
            goal.type = "detect"
            goal.obj.location = location
        elif perception_task == PerceptionTask.OBJECT_DETECTION2D:
            goal.type = "detect"
        elif perception_task == PerceptionTask.HUMAN_TRACKING:
            goal.type = "track"
        elif perception_task == PerceptionTask.FREE_SPACE:
            goal.type = "detect"
            goal.obj.attribute.append("free")
            goal.obj.location = location
        else:
            # Resort basically to object recogntion 2d
            goal.type = "detect"

        self.client.send_goal(goal)

    def abort_goal(self):
        """
        Abort the currently running action.
        """
        self.client.cancel_goal()

    def handle_result(self) -> robokudo_msgs.msg.QueryResult:
        """
        Wait for the result from the action server and handle it.
        """
        self.client.wait_for_result()
        result = self.client.get_result()
        # Handle the result (you can customize this part based on your needs)
        rospy.loginfo(f"Received result: {result}")  # Adjust this based on your actual result message structure
        return result

# https://code-with-me.global.jetbrains.com/hpxgM92M7ll5zM1diHf1qw#p=PY&fp=33941BA9C6BFF38413D50109535838B64587A7952FCF9BC2D7E883808E7DAE7A&newUi=true

class CRAM:
    def __init__(self):
        self.objbelief_state: Dict[str, ObjectEntry] = {}
        self.giskard = GiskardWrapper()
        self.gripper = Gripper()
        self.rk_client = QueryActionClient()
        self.talker_pub = rospy.Publisher('/talk_request_action/goal', TalkRequestActionGoal, queue_size=10)
        self.tip_link = 'hand_gripper_tool_frame'
        self.map = 'map'
        self.fancylogger = Logger("/home/toya/rkop_experiments/")
        self.grasped_uuid = None

    def reset(self):
        self.gripper.open()
        self.giskard.clear_world()
        table_pose = PoseStamped()
        table_pose.header.frame_id = self.map
        table_pose.pose.position.x = 12.84
        table_pose.pose.position.y = 2.96
        table_pose.pose.position.z = 0.205
        self.giskard.add_box(name='table',
                             size=(1.3, 0.8, 0.405),
                             pose=table_pose,
                             parent_link=self.map)
        self.giskard.set_joint_goal(default_pose)
        self.giskard.plan_and_execute()
        base_pose = PoseStamped()
        base_pose.header.frame_id = self.map
        base_pose.pose.position = Point(12.008, 1.939, 0.000)
        base_pose.pose.orientation = Quaternion(0.000, 0.000, 0.518, 0.855)
        self.giskard.set_cart_goal(goal_pose=base_pose,
                                   tip_link='base_footprint',
                                   root_link=self.map)
        self.giskard.plan_and_execute()

    def detect_objects_on_table(self):
        self.giskard.set_joint_goal(detect_pose)
        self.giskard.plan_and_execute()
        hsr_say(self.talker_pub, "Welcome back home. Let me take a look at the new objects you brought.")
        self.fancylogger.log_event("crampylog.csv", "object_detection3d", "START")
        # client.send_goal(PerceptionTask.OBJECT_DETECTION2D)
        self.rk_client.send_goal(PerceptionTask.OBJECT_DETECTION3D,location="sofa_table")
        rk_result = self.rk_client.handle_result()
        self.fancylogger.log_event("crampylog.csv", "object_detection3d", "END")

        if len(rk_result.res) == 0:
            print("No objects found by RK")
        else:
            print(f"Found {len(rk_result.res)} object(s)")

        detected_object: robokudo_msgs.msg.ObjectDesignator = None
        for detected_object in rk_result.res:
            oe = ObjectEntry(pose=detected_object.pose[0], classname="", size=np.array([
                detected_object.shape_size[0].dimensions.x,
                detected_object.shape_size[0].dimensions.y,
                detected_object.shape_size[0].dimensions.z,
            ]), object_id=detected_object.uid)
            self.objbelief_state[detected_object.uid] = oe
            self.giskard.add_box(name=detected_object.uid,
                            size=(oe.size[0], oe.size[1], oe.size[2]),
                            pose=detected_object.pose[0],
                            parent_link=self.map)

    def detect_human_pointing(self):
        self.fancylogger.log_event("crampylog.csv", "human_pointing", "START")
        self.rk_client.send_goal(PerceptionTask.HUMAN_POINTING)
        rk_result = self.rk_client.handle_result()
        self.fancylogger.log_event("crampylog.csv", "human_pointing", "END")
        assert(len(rk_result.res)>0)
        obj_pointed_at = rk_result.res[0].uid
        return obj_pointed_at

    def trigger_human_tracking(self):
        self.fancylogger.log_event("crampylog.csv", "human_tracking", "START")
        self.rk_client.send_goal(PerceptionTask.HUMAN_TRACKING)

    def abort_human_tracking(self):
        rk_result = self.rk_client.abort_goal()
        self.fancylogger.log_event("crampylog.csv", "human_tracking", "END")

    def grasp_object_from_table(self, uuid: str):
        self.fancylogger.log_event("crampylog.csv", "grasp_object_from_table", "START")
        self.gripper.open()
        object_to_grasp = self.objbelief_state[uuid]
        map_P_center = np.zeros(3)
        for uuid, object_entry in self.objbelief_state.items():
            map_P_center += object_entry.map_P_object
        map_P_center /= len(self.objbelief_state)

        direction = object_to_grasp.map_P_object - map_P_center
        approach_direction_v = object_to_grasp.map_P_object + direction

        approach_direction = PointStamped()
        approach_direction.header.frame_id = self.map
        approach_direction.point.x = approach_direction_v[0]
        approach_direction.point.y = approach_direction_v[1]
        approach_direction.point.z = approach_direction_v[2]

        self.giskard.set_joint_goal(default_pose)
        self.giskard.plan_and_execute()
        # giskard.set_json_goal('GraspBox',
        #                       UUID=first_object.uuid,
        #                       tip_link=tip_link,
        #                       root_link=map,
        #                       approach_direction=approach_direction)
        self.giskard.set_json_goal('GraspBoxMalte',
                              object_name=object_to_grasp.uuid,
                              root_link=self.map,
                              approach_hint=approach_direction)
        self.giskard.allow_collision('hsrb', object_to_grasp.uuid)
        self.giskard.plan()
        self.gripper.close()
        self.grasped_uuid = object_to_grasp.uuid
        self.giskard.update_parent_link_of_group(name=self.grasped_uuid,
                                                 parent_link=self.tip_link)
        # %% lift
        gripper_goal = PoseStamped()
        gripper_goal.header.frame_id = self.tip_link
        gripper_goal.pose.position.x = 0.15
        gripper_goal.pose.orientation.w = 1
        self.giskard.set_cart_goal(gripper_goal, tip_link=self.tip_link, root_link=self.map)
        self.giskard.plan_and_execute()

        self.giskard.set_joint_goal(carry_pose)
        self.giskard.plan_and_execute()

        self.fancylogger.log_event("crampylog.csv", "grasp_object_from_table", "END")

    def point_at_sofa(self, height: float = 0.75):
        goal_point = PointStamped()
        goal_point.header.frame_id = self.map
        goal_point.point.x = 13.711359024047852
        goal_point.point.y = 4.189180374145508
        goal_point.point.z = height
        self.point_at(goal_point)

    def point_at(self, goal_point: PointStamped):
        head = 'head_rgbd_sensor_link'
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = head
        pointing_axis.vector.z = 1
        self.giskard.set_pointing_goal(goal_point=goal_point,
                                       tip_link=head,
                                       root_link=self.map,
                                       pointing_axis=pointing_axis)
        self.giskard.plan_and_execute()

    def carry_my_bs(self):
        self.giskard.set_json_goal('CarryMyBullshit',
                                   patrick_topic_name='robokudo/human_position')
        self.giskard.allow_all_collisions()
        self.giskard.set_json_goal('EndlessMode')
        self.giskard.plan_and_execute()

    def place_carried_object(self):
        pass

    def end_logging(self):
        self.fancylogger.log_event("crampylog.csv", "experiment", "END")



rospy.init_node('cram')
tf.init(10)

cram = CRAM()
cram.reset()
cram.fancylogger.log_event("crampylog.csv", "experiment", "START")
cram.detect_objects_on_table()
cram.point_at_sofa()
uuid = cram.detect_human_pointing()
cram.grasp_object_from_table(uuid)
cram.point_at_sofa()
hsr_say(cram.talker_pub,"I'll now follow you")
cram.trigger_human_tracking()
cram.carry_my_bs()



cram.fancylogger.log_event("crampylog.csv", "move_start_pose", "START")






cram.fancylogger.log_event("crampylog.csv", "move_start_pose","END")



cram.fancylogger.log_event("crampylog.csv", "experiment", "END")
