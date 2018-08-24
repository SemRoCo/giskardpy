import hashlib

from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_matrix

import symengine_wrappers as sw
from giskardpy import BACKEND
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import Plugin
from giskardpy.symengine_robot import Robot
from giskardpy.utils import keydefaultdict, urdfs_equal


class RobotPlugin(Plugin):
    """
    Efficiently keeps a symengine robot in sync with the god map.
    Inherit from this plugin if you want to use symengine robots who's urdf is in the god map.
    """

    def __init__(self, robot_description_identifier, js_identifier, default_joint_vel_limit=0):
        """
        :type robot_description_identifier: str
        :type js_identifier: str
        :type default_joint_vel_limit: float
        """
        self._robot_description_identifier = robot_description_identifier
        self._joint_states_identifier = js_identifier
        self.default_joint_vel_limit = default_joint_vel_limit
        self.robot = None
        self.__urdf_updated = False
        super(RobotPlugin, self).__init__()

    def start_always(self):
        if self.__is_urdf_updated():
            self.init_robot()
            self.__urdf_updated = True
        else:
            self.__urdf_updated = False

    def __is_urdf_updated(self):
        new_urdf = self.god_map.get_data([self._robot_description_identifier])
        # TODO figure out a better solution which does not require the urdf to be rehashed all the time
        return self.get_robot() is None or not urdfs_equal(self.get_robot().get_urdf(), new_urdf)

    def was_urdf_updated(self):
        return self.__urdf_updated

    def init_robot(self):
        urdf = self.god_map.get_data([self._robot_description_identifier])
        self.robot = Robot(urdf, self.default_joint_vel_limit)
        current_joints = JointStatesInput(self.god_map.to_symbol,
                                          self.get_robot().get_joint_names_controllable(),
                                          [self._joint_states_identifier],
                                          [u'position'])
        self.get_robot().parse_urdf(current_joints.joint_map)

    def get_robot(self):
        """
        :rtype: Robot
        """
        return self.robot


class FKPlugin(RobotPlugin):
    """
    Puts all forward kinematics of a robot in the god map, but they are only computed on demand.
    """
    def __init__(self, fk_identifier, js_identifier, robot_description_identifier):
        self.fk_identifier = fk_identifier
        self.fk = None
        self.robot = None
        super(FKPlugin, self).__init__(robot_description_identifier, js_identifier)

    def update(self):
        exprs = self.god_map.get_symbol_map()

        def on_demand_fk_evaluated(key):
            """
            :param key: (root_name, tip_name)
            :type key: tuple
            :rtype: PoseStamped
            """
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

    def start_always(self):
        super(FKPlugin, self).start_always()
        if self.was_urdf_updated():
            free_symbols = self.god_map.get_registered_symbols()

            def on_demand_fk(key):
                """
                :param key: (root_name, tip_name)
                :type key: tuple
                :return: function that takes a expression dict and returns the fk
                """
                # TODO possible speed up by merging fks into one matrix
                root, tip = key
                fk = self.robot.get_fk_expression(root, tip)
                return sw.speed_up(fk, free_symbols, backend=BACKEND)

            self.fk = keydefaultdict(on_demand_fk)

    def stop(self):
        pass

    def copy(self):
        cp = self.__class__(self.fk_identifier, self._joint_states_identifier, self._robot_description_identifier)
        cp.fk = self.fk
        cp.robot = self.robot
        return cp
