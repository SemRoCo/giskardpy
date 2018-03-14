import rospy

from giskardpy.application import ROSApplication
from giskardpy.plugin_joint_state import JointStatePlugin
from giskardpy.process_manager import ProcessManager
from giskardpy.plugin_pybullet import PyBullet

if __name__ == '__main__':
    rospy.init_node('muh')

    js_plugin = JointStatePlugin()
    pb = PyBullet()

    process_manager = ProcessManager()
    process_manager.register_plugin('a', js_plugin)
    process_manager.register_plugin('c', pb)

    app = ROSApplication(process_manager)
    app.run()
