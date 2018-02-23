import rospy

from giskardpy.application import ROSApplication
from giskardpy.joint_state import JointStateInput
from giskardpy.print_joint_state import PrintJointState
from giskardpy.process_manager import ProcessManager

if __name__ == '__main__':
    rospy.init_node('muh')

    js_plugin = JointStateInput()
    printer = PrintJointState()

    process_manager = ProcessManager()
    process_manager.register_plugin('a', js_plugin)
    process_manager.register_plugin('b', printer)

    app = ROSApplication(process_manager)

    rospy.spin()
