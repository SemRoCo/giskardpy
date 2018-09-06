from giskardpy.input_system import JointStatesInput
from giskardpy.plugin import NewPluginBase, GiskardBehavior
from giskardpy.symengine_robot import Robot
from giskardpy.utils import urdfs_equal


class NewRobotPlugin(NewPluginBase):

    def __init__(self, robot_description_identifier, js_identifier, default_joint_vel_limit=0):
        """
        :type robot_description_identifier: str
        :type js_identifier: str
        :type default_joint_vel_limit: float
        """
        super(NewRobotPlugin, self).__init__()
        self._robot_description_identifier = robot_description_identifier
        self._joint_states_identifier = js_identifier
        self.default_joint_vel_limit = default_joint_vel_limit
        self.robot = None
        self.__urdf_updated = False
        self.controlled_joints = set()
        self.controllable_links = set()
        self.init_robot()

    def initialize(self):
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

    def update_controlled_joints_and_links(self, controlled_joints_identifier, controllable_links_identifier):
        """
        Gets controlled joints from god map and uses this to calculate the controllable link, which are written to
        the god map.
        """
        self.controlled_joints = self.god_map.get_data([controlled_joints_identifier])
        self.controllable_links = set()
        for joint_name in self.controlled_joints:
            self.controllable_links.update(self.get_robot().get_sub_tree_link_names_with_collision(joint_name))
        self.god_map.set_data([controllable_links_identifier], self.controllable_links)



# class RobotBehavior(GiskardBehavior):