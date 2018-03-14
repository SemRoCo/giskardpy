from geometry_msgs.msg import PoseStamped

from giskardpy.plugin import IOPlugin
from giskardpy.tfwrapper import TfWrapper


class FKPlugin(IOPlugin):
    def __init__(self, root, tip):
        self.root = root
        self.tip = tip
        super(FKPlugin, self).__init__()

    def get_readings(self):
        p = PoseStamped()
        p.header.frame_id = self.tip
        p.pose.orientation.w = 1
        return {'fk_{}'.format(self.tip): self.tf.transform_pose(self.root, p)}

    def update(self):
        super(FKPlugin, self).update()

    def start(self, god_map):
        self.tf = TfWrapper()
        super(FKPlugin, self).start(god_map)

    def stop(self):
        pass

    def copy(self):
        return self



# import rospy
# from geometry_msgs.msg import PoseStamped, Quaternion
# from tf.transformations import quaternion_from_matrix
#
# import symengine_wrappers as sw
# from giskardpy.input_system import JointStatesInput
# from giskardpy.plugin import IOPlugin
# from giskardpy.robot_constraints import Robot
#
#
# class FKPlugin(IOPlugin):
#     def __init__(self, root, tip):
#         self.root = root
#         self.tip = tip
#         self._joint_states_identifier = 'js'
#         super(FKPlugin, self).__init__()
#
#     def get_readings(self):
#         fk = self.fk(**self.databus.get_expr_values())
#         p = PoseStamped()
#         p.header.frame_id = self.tip
#         p.pose.position.x = sw.pos_of(fk)[0,0]
#         p.pose.position.y = sw.pos_of(fk)[1,0]
#         p.pose.position.z = sw.pos_of(fk)[2,0]
#         orientation = quaternion_from_matrix(fk)
#         p.pose.orientation = Quaternion(*orientation)
#         return {'fk_{}'.format(self.tip): p}
#
#     def update(self):
#         super(FKPlugin, self).update()
#
#     def start(self, databus):
#         super(FKPlugin, self).start(databus)
#         urdf = rospy.get_param('robot_description')
#         self.robot = Robot(urdf)
#         current_joints = JointStatesInput.prefix_constructor(self.databus.get_expr,
#                                                              self.robot.get_joint_names(),
#                                                              self._joint_states_identifier,
#                                                              'position')
#         self.robot.set_joint_symbol_map(current_joints)
#         fk = self.robot.get_fk_expression(self.root, self.tip)
#         self.fk = sw.speed_up(fk, fk.free_symbols)
#
#     def stop(self):
#         pass

