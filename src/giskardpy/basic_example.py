#!/usr/bin/env python
import sys
import rospy
from sensor_msgs.msg import JointState

from giskardpy.robot import Robot
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.input_system import *
from giskardpy.symengine_wrappers import *

class MyPositionController(QPController):
	def __init__(self, robot, eef_name):
		self.eef_name = eef_name
		super(MyPositionController, self).__init__(robot)

	def add_inputs(self, robot):
		self.position_input = Point3Input('goal')

	def make_constraints(self, robot):
		super(MyPositionController, self).make_constraints(robot)

		d = norm(self.position_input.get_expression() - pos_of(robot.frames[self.eef_name]))
		self._soft_constraints['position constraint'] = SoftConstraint(-d, -d, 1, d)

	def set_goal(self, x, y, z):
		self.update_observables(self.position_input.get_update_dict(x, y, z))


class MyPositionControllerNode(object):
	def __init__(self, robot_file, eef_name, x, y, z):
		self.robot = Robot()
		self.robot.load_from_urdf_path(robot_file, 'base_link', [eef_name])
		self.controller = MyPositionController(self.robot, eef_name)
		self.controller.set_goal(x,y,z)
		self.cmd_pub = rospy.Publisher('simulator/commands', JointState, queue_size=1)
		self.cmd_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)

	def js_callback(self, joint_state):
		js_dict = {joint_state.name[x]: joint_state.position[x] for x in range(len(joint_state.name))}
		self.robot.set_joint_state(js_dict)
		cmd = self.controller.get_next_command()

		cmd_msg = JointState()
		cmd_msg.header.stamp = rospy.Time.now()
		for joint_name, velocity in cmd.items():
			cmd_msg.name.append(joint_name)
			cmd_msg.velocity.append(velocity)
			cmd_msg.position.append(0)
			cmd_msg.effort.append(0)

		self.cmd_pub.publish(cmd_msg)


if __name__ == '__main__':
	if len(sys.argv) < 6:
		print('Please provide: <urdf file> <endeffector name> <goal x> <goal y> <goal z>')
		exit(0)

	rospy.init_node('basic_eef_position_controller')

	node = MyPositionControllerNode(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))

	rospy.spin()
