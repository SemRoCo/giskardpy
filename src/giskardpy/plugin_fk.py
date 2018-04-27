import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_matrix

import symengine_wrappers as sw
from giskardpy import BACKEND
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import Plugin
from giskardpy.symengine_robot import Robot
from giskardpy.utils import keydefaultdict


class FKPlugin(Plugin):
    def __init__(self, roots, tips, fk_identifier='fk', js_identifier='js'):
        if len(roots) != len(tips):
            # TODO look for better exception
            raise Exception('lengths of roots, tips and fk_identifiers don\'t match')
        self.roots = roots
        self.tips = tips
        self._joint_states_identifier = js_identifier
        self.fk_identifier = fk_identifier
        self.fk = None
        super(FKPlugin, self).__init__()

    def get_readings(self):
        exprs = self.god_map.get_expr_values()
        def on_demand_fk_evaluated(key):
            root, tip = key
            fk = self.fk[root, tip](**exprs)
            p = PoseStamped()
            p.header.frame_id = tip
            p.pose.position.x = sw.pos_of(fk)[0, 0]
            p.pose.position.y = sw.pos_of(fk)[1, 0]
            p.pose.position.z = sw.pos_of(fk)[2, 0]
            orientation = quaternion_from_matrix(fk)
            p.pose.orientation = Quaternion(*orientation)
            return p
        fks = keydefaultdict(on_demand_fk_evaluated)
        return {self.fk_identifier: fks}

    def update(self):
        super(FKPlugin, self).update()

    def start(self, god_map):
        super(FKPlugin, self).start(god_map)
        if self.fk is None:
            urdf = rospy.get_param('robot_description')
            self.robot = Robot(urdf)
            joint_names = self.robot.get_joint_names()
            current_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                                 joint_names,
                                                                 self._joint_states_identifier,
                                                                 'position')
            self.robot.set_joint_symbol_map(current_joints)

            free_symbols = self.god_map.get_free_symbols()
            def on_demand_fk(key):
                # TODO possible speed up by merging fks into one matrix
                root, tip = key
                fk = self.robot.get_fk_expression(root, tip)
                return sw.speed_up(fk, free_symbols, backend=BACKEND)

            self.fk = keydefaultdict(on_demand_fk)

    def stop(self):
        pass

    def get_replacement_parallel_universe(self):
        return self
