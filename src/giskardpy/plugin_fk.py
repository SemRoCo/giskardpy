import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_matrix

import symengine_wrappers as sw
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import Plugin
from giskardpy.symengine_robot import Robot


class FKPlugin(Plugin):
    def __init__(self, root, tip):
        self.root = root
        self.tip = tip
        self._joint_states_identifier = 'js'
        super(FKPlugin, self).__init__()

    def get_readings(self):
        fk = self.fk(**self.god_map.get_expr_values())
        p = PoseStamped()
        p.header.frame_id = self.tip
        p.pose.position.x = sw.pos_of(fk)[0,0]
        p.pose.position.y = sw.pos_of(fk)[1,0]
        p.pose.position.z = sw.pos_of(fk)[2,0]
        orientation = quaternion_from_matrix(fk)
        p.pose.orientation = Quaternion(*orientation)
        return {'fk_{}'.format(self.tip): p}

    def update(self):
        super(FKPlugin, self).update()

    def start(self, god_map):
        super(FKPlugin, self).start(god_map)
        urdf = rospy.get_param('robot_description')
        self.robot = Robot(urdf)
        current_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                             self.robot.get_chain_joints(self.root, self.tip),
                                                             self._joint_states_identifier,
                                                             'position')
        self.robot.set_joint_symbol_map(current_joints)
        fk = self.robot.get_fk_expression(self.root, self.tip)
        self.fk = sw.speed_up(fk, fk.free_symbols)

    def stop(self):
        pass

    def copy(self):
        return self

