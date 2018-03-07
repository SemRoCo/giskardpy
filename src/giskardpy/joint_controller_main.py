import rospy

from giskardpy.action_server_plugin import ActionServer
from giskardpy.application import ROSApplication
from giskardpy.controller_plugin import JointControllerPlugin
from giskardpy.joint_state import JointStateInput
from giskardpy.process_manager import ProcessManager
from giskardpy.pybullet_plugin import PyBullet

if __name__ == '__main__':
    rospy.init_node('muh')

    js_plugin = JointStateInput()
    jc = JointControllerPlugin()
    action_server = ActionServer()

    process_manager = ProcessManager()
    process_manager.register_plugin('a', js_plugin)
    process_manager.register_plugin('c', action_server)
    process_manager.register_plugin('b', jc)

    app = ROSApplication(process_manager)
    app.run()
