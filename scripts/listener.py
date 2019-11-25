#!/usr/bin/env python

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
import rospy
import message_filters as MF
from tf2_msgs.msg import TFMessage
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState

class Listener:
  """
  This class help to get the current value of joints
  """
  _topic= ""
  _typ_msg= ""
  _msg= None
  def __init__(self):
    print("Listener of topic is actived")

  def set_topic_and_typMEssage(self, topic, typ_msg):
    """
    this method set the topic and the typ of msg to listen
    :param topic: str
    :param typ_msg: Object
    :return:
    """
    self._topic= topic
    self._typ_msg = typ_msg

  def listen_topic(self):
    """
    the method listens the topic
    :return:
    """
    self._msg = rospy.wait_for_message(self._topic, self._typ_msg)
    print (self._msg.joint_names)
    print (self._msg.actual.positions)

  def listen_topic_with_sensor_msg(self):
    """
    this method listens the topic
    :return:
    """
    self._msg = rospy.wait_for_message(self._topic, self._typ_msg)
    print(self._msg.name)
    print(self._msg.position)

  def get_value(self, joint_name):
    """
    this method get the value of special joint
    :param joint_name: str
    :return: float
    """
    for x in range(0, len(self._msg.joint_names)):  # for x in range(0, len(self._msg.joint_names)):
      if self._msg.joint_names[x] == joint_name:
        return round(self._msg.actual.positions[x], 2)
    return None

  def get_value_from_sensor_msg(self, joint_name):
    """
    this method get value of joint from sensor msg
    :param joint_name: str
    :return: float
    """
    for x in range(0, len(self._msg.name)):  # for x in range(0, len(self._msg.joint_names)):
      if self._msg.name[x] == joint_name:
        return round(self._msg.position[x], 2)
    return None

if __name__ == '__main__':
  rospy.init_node('check_tf')
  l= Listener()
  #l.set_topic_and_typMEssage("/whole_body_controller/state", JointTrajectoryControllerState)
  #l.set_topic_and_typMEssage("/hsrb/omni_base_controller/state", JointTrajectoryControllerState)
  l.set_topic_and_typMEssage("/kitchen/joint_states", JointState) #("/hsrb/joint_states", JointState)
  #l.listen_topic()
  #print l.get_value("")
  l.listen_topic_with_sensor_msg()
  print(l.get_value_from_sensor_msg("iai_kitchen/iai_fridge_door_handle")) #("hand_motor_joint")
  #print l.get_value("odom_y")
  #print l.get_value("odom_t")
  #print("is done")
  
