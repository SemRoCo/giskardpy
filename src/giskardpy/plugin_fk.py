import hashlib

from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_matrix

import symengine_wrappers as sw
from giskardpy import BACKEND
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import Plugin
from giskardpy.symengine_robot import Robot
from giskardpy.utils import keydefaultdict, urdfs_equal


class FKPlugin(Plugin):
    def __init__(self, fk_identifier, js_identifier, robot_description_identifier):
        self._joint_states_identifier = js_identifier
        self.robot_description_identifier = robot_description_identifier
        self.fk_identifier = fk_identifier
        self.fk = None
        self.robot = None
        super(FKPlugin, self).__init__()

    def update(self):
        # TODO don't use start once here
        self.start_once()
        exprs = self.god_map.get_symbol_map()

        def on_demand_fk_evaluated(key):
            fk = self.fk[key](**exprs)
            p = PoseStamped()
            p.header.frame_id = key[1]
            p.pose.position.x = sw.position_of(fk)[0, 0]
            p.pose.position.y = sw.position_of(fk)[1, 0]
            p.pose.position.z = sw.position_of(fk)[2, 0]
            p.pose.orientation = Quaternion(*quaternion_from_matrix(fk))
            return p

        fks = keydefaultdict(on_demand_fk_evaluated)
        self.god_map.set_data([self.fk_identifier], fks)

    def start_once(self):
        new_urdf = self.god_map.get_data([self.robot_description_identifier])
        if self.get_robot() is None or urdfs_equal(self.get_robot().get_urdf(), new_urdf):
            self.robot = Robot(new_urdf)
            joint_names = self.robot.get_joint_names_controllable()
            current_joints = JointStatesInput(self.god_map.to_symbol,
                                              joint_names,
                                              (self._joint_states_identifier,),
                                              ('position',))
            self.robot.parse_urdf(current_joints.joint_map)

            free_symbols = self.god_map.get_registered_symbols()

            def on_demand_fk(key):
                # TODO possible speed up by merging fks into one matrix
                root, tip = key
                fk = self.robot.get_fk_expression(root, tip)
                return sw.speed_up(fk, free_symbols, backend=BACKEND)

            self.fk = keydefaultdict(on_demand_fk)

    def get_robot(self):
        """
        :rtype: Robot
        """
        return self.robot

    def stop(self):
        pass

    def copy(self):
        cp = self.__class__(self.fk_identifier, self._joint_states_identifier, self.robot_description_identifier)
        cp.fk = self.fk
        cp.robot = self.robot
        return cp
