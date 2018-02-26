import rospy

from giskardpy.application import ROSApplication
from giskardpy.joint_state import JointStateInput
from giskardpy.process_manager import ProcessManager
from giskardpy.pybullet_plugin import PyBullet

if __name__ == '__main__':
    rospy.init_node('muh')

    js_plugin = JointStateInput()
    pb = PyBullet()

    process_manager = ProcessManager()
    process_manager.register_plugin('a', js_plugin)
    process_manager.register_plugin('c', pb)

    app = ROSApplication(process_manager)

    rospy.spin()
