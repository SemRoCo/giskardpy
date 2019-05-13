#!/usr/bin/env python
import rospy
from actionlib import SimpleActionClient
from giskard_msgs.msg import MoveAction, MoveGoal, MoveCmd, Controller
from giskardpy import logging

if __name__ == '__main__':
    rospy.init_node('init_chain')
    client = SimpleActionClient('qp_controller/command', MoveAction)
    client.wait_for_server()
    roots = rospy.get_param('~roots')
    tips = rospy.get_param('~tips')
    typess = rospy.get_param('~types')
    goal = MoveGoal()
    move_cmd = MoveCmd()
    if not (len(roots) == len(tips) and len(tips) == len(typess)):
        raise Exception('number of roots, tips and types not equal')
    for root, tip, types in zip(roots, tips, typess):
        for type in types:
            controller = Controller()
            controller.root_link = root
            controller.tip_link = tip
            controller.type = type
            controller.weight = 0
            controller.goal_pose.pose.orientation.w = 1
            controller.goal_pose.header.frame_id = tip
            move_cmd.controllers.append(controller)
    goal.type = MoveGoal.PLAN_ONLY
    goal.cmd_seq.append(move_cmd)
    logging.loginfo('sending goal')
    client.send_goal_and_wait(goal)