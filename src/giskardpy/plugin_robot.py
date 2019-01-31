from giskardpy.identifier import robot_description_identifier, default_joint_weight_identifier, js_identifier, \
    robot_identifier
from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import PluginBase
from giskardpy.symengine_robot import Robot
from giskardpy.utils import urdfs_equal


class RobotPlugin(PluginBase):

    def __init__(self, default_joint_vel_limit=None, default_joint_weight=None):
        """
        :type robot_description_identifier: str
        :type js_identifier: str
        :type default_joint_vel_limit: float
        """
        super(RobotPlugin, self).__init__()
        if not isinstance(self.get_robot(), Robot):
            if default_joint_vel_limit is None and \
                default_joint_weight is None:
                raise Exception(u'Bug in your code: robot not initialized, pls call with parameters.')
            self.default_joint_vel_limit = default_joint_vel_limit
            self.default_joint_weight = default_joint_weight
            self.__urdf_updated = True
            self.controlled_joints = set()
            self.controllable_links = set()

    def get_god_map(self):
        """
        :rtype: giskardpy.god_map.GodMap
        """
        return self.god_map

    def initialize(self):
        if self.__is_urdf_updated():
            self.init_robot()
            self.__urdf_updated = True
        else:
            self.__urdf_updated = False

    def __is_urdf_updated(self):
        new_urdf = self.god_map.safe_get_data([robot_description_identifier])
        # TODO figure out a better solution which does not require the urdf to be rehashed all the time
        return self.get_robot() is None or not urdfs_equal(self.get_robot().get_urdf(), new_urdf)

    def was_urdf_updated(self):
        return self.__urdf_updated

    def init_robot(self):
        urdf = self.god_map.safe_get_data([robot_description_identifier])
        if default_joint_weight_identifier is not None:
            default_joint_weight = self.get_god_map().to_symbol([default_joint_weight_identifier])
            self.get_god_map().safe_set_data([default_joint_weight_identifier], self.default_joint_weight)
        else:
            default_joint_weight = self.default_joint_weight
        self.get_god_map().safe_set_data([robot_identifier], Robot(urdf, self.default_joint_vel_limit, default_joint_weight))
        current_joints = JointStatesInput(self.god_map.to_symbol,
                                          self.get_robot().get_joint_names_controllable(),
                                          [js_identifier],
                                          [u'position'])
        self.get_robot().parse_urdf(current_joints.joint_map)

    def get_robot(self):
        """
        :rtype: Robot
        """
        return self.get_god_map().safe_get_data([robot_identifier])

    def update_controlled_joints_and_links(self, controlled_joints_identifier, controllable_links_identifier=None):
        """
        Gets controlled joints from god map and uses this to calculate the controllable link, which are written to
        the god map.
        """
        self.controlled_joints = self.god_map.safe_get_data([controlled_joints_identifier])
        self.controllable_links = set()
        if controllable_links_identifier is not None:
            for joint_name in self.controlled_joints:
                self.controllable_links.update(self.get_robot().get_sub_tree_link_names_with_collision(joint_name))
            self.god_map.safe_set_data([controllable_links_identifier], self.controllable_links)



class RobotKinPlugin(RobotPlugin):
    def __init__(self):
        super(RobotKinPlugin, self).__init__(0, 0)
