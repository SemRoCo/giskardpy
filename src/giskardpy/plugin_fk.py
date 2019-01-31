from geometry_msgs.msg import PoseStamped, Quaternion
from py_trees import Status
from tf.transformations import quaternion_from_matrix

import symengine_wrappers as sw
from giskardpy import BACKEND
from giskardpy.identifier import fk_identifier
from giskardpy.plugin_robot import RobotPlugin, RobotKinPlugin
from giskardpy.utils import keydefaultdict


class FkPlugin(RobotKinPlugin):
    def __init__(self):
        self.fk = None
        self.robot = None
        super(FkPlugin, self).__init__()

    def initialize(self):
        super(FkPlugin, self).initialize()
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
        self.god_map.safe_set_data([fk_identifier], fks)
        return None