import rospy

from giskardpy.action_server_plugin import ActionServer
from giskardpy.application import ROSApplication
from giskardpy.controller_plugin import JointControllerPlugin, CartesianControllerPlugin
from giskardpy.fk_plugin import FKPlugin
from giskardpy.joint_state import JointStatePlugin
from giskardpy.process_manager import ProcessManager
from giskardpy.pybullet_plugin import PyBullet

if __name__ == '__main__':
    rospy.init_node('muh')

    process_manager = ProcessManager()
    process_manager.register_plugin('a', JointStatePlugin())
    process_manager.register_plugin('fk', FKPlugin('base_footprint', 'gripper_tool_frame'))
    process_manager.register_plugin('b', ActionServer())
    process_manager.register_plugin('c', JointControllerPlugin())
    process_manager.register_plugin('d', CartesianControllerPlugin())

    app = ROSApplication(process_manager)
    app.run()
