from geometry_msgs.msg import PoseStamped, Quaternion
from py_trees import Status
from tf.transformations import quaternion_from_matrix

import symengine_wrappers as sw
from giskardpy import BACKEND
from giskardpy.plugin_robot import NewRobotPlugin
from giskardpy.utils import keydefaultdict


class NewFkPlugin(NewRobotPlugin):
    def __init__(self, fk_identifier, js_identifier, robot_description_identifier):
        self.fk_identifier = fk_identifier
        self.fk = None
        self.robot = None
        super(NewFkPlugin, self).__init__(robot_description_identifier, js_identifier)

    def initialize(self):
        super(NewFkPlugin, self).initialize()
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
        self.god_map.safe_set_data([self.fk_identifier], fks)
        return None