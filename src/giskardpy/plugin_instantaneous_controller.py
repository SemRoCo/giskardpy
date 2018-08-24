from giskard_msgs.msg import Controller

from giskardpy.input_system import FrameInput, Point3Input, Vector3Input, \
    ShortestAngularDistanceInput
from giskardpy.plugin_fk import RobotPlugin
from giskardpy.symengine_controller import SymEngineController, position_conv, rotation_conv, \
    link_to_link_avoidance, joint_position, continuous_joint_position
import symengine_wrappers as sw


class CartesianBulletControllerPlugin(RobotPlugin):
    """
    Instantaneous controller that can do joint/cartesian movements while avoiding collisions.
    """

    def __init__(self, root_link, js_identifier, fk_identifier, goal_identifier, next_cmd_identifier,
                 collision_identifier, closest_point_identifier, controlled_joints_identifier,
                 controllable_links_identifier, robot_description_identifier,
                 collision_goal_identifier, pyfunction_identifier, path_to_functions, nWSR, default_joint_vel_limit):
        """
        :param root_link: the robots root link
        :type root_link: str
        :type js_identifier: str
        :type fk_identifier: str
        :type goal_identifier: str
        :type next_cmd_identifier: str
        :type collision_identifier:  str
        :type closest_point_identifier: str
        :type controlled_joints_identifier: str
        :type controllable_links_identifier: str
        :type robot_description_identifier: str
        :type collision_goal_identifier: str
        :type pyfunction_identifier: str
        :param path_to_functions: path to folder where compiled functions and the self collision matrix are stored
        :type path_to_functions: str
        :param nWSR: magic number for QP solver. has to be big for difficult problems. choose None if you're like wtf.
        :type nWSR: Union[int, None]
        :param default_joint_vel_limit: caps the joint velocities defined in the urdf.
        :type default_joint_vel_limit: float
        """
        self.collision_goal_identifier = collision_goal_identifier
        self.controlled_joints_identifier = controlled_joints_identifier
        self._pyfunctions_identifier = pyfunction_identifier
        self._fk_identifier = fk_identifier
        self._collision_identifier = collision_identifier
        self._closest_point_identifier = closest_point_identifier
        self.path_to_functions = path_to_functions
        self.nWSR = nWSR
        self.default_joint_vel_limit = default_joint_vel_limit
        self.root = root_link
        self._joint_states_identifier = js_identifier
        self._goal_identifier = goal_identifier
        self._next_cmd_identifier = next_cmd_identifier
        self.controllable_links_identifier = controllable_links_identifier
        self.controller = None
        self.known_constraints = set()
        self.controlled_joints = set()
        self.max_number_collision_entries = 10
        self.robot = None
        self.used_joints = set()
        self.controllable_links = set()
        super(CartesianBulletControllerPlugin, self).__init__(robot_description_identifier,
                                                              self._joint_states_identifier,
                                                              self.default_joint_vel_limit)

    def copy(self):
        cp = self.__class__(self.root, self._joint_states_identifier, self._fk_identifier,
                            self._goal_identifier, self._next_cmd_identifier, self._collision_identifier,
                            self._closest_point_identifier, self.controlled_joints_identifier,
                            self.controllable_links_identifier, self._robot_description_identifier,
                            self.collision_goal_identifier, self._pyfunctions_identifier,
                            self.path_to_functions, self.nWSR,
                            self.default_joint_vel_limit)
        cp.controller = self.controller
        cp.robot = self.robot
        cp.known_constraints = self.known_constraints
        cp.controlled_joints = self.controlled_joints
        return cp

    def start_always(self):
        super(CartesianBulletControllerPlugin, self).start_always()
        self.next_cmd = {}
        self.update_controlled_joints_and_links()
        if self.was_urdf_updated():
            self.init_controller()
            self.add_js_controller_soft_constraints()
            self.add_collision_avoidance_soft_constraints()
        self.add_cart_controller_soft_constraints()
        self.set_unused_joint_goals_to_current()

    def update(self):
        expr = self.god_map.get_symbol_map()
        next_cmd = self.controller.get_cmd(expr, self.nWSR)
        self.next_cmd.update(next_cmd)

        self.god_map.set_data([self._next_cmd_identifier], self.next_cmd)

    def update_controlled_joints_and_links(self):
        self.controlled_joints = self.god_map.get_data([self.controlled_joints_identifier])
        self.controllable_links = set()
        for joint_name in self.controlled_joints:
            self.controllable_links.update(self.get_robot().get_sub_tree_link_names_with_collision(joint_name))
        self.god_map.set_data([self.controllable_links_identifier], self.controllable_links)

    def init_controller(self):
        self.controller = SymEngineController(self.robot, self.path_to_functions)
        self.controller.set_controlled_joints(self.controlled_joints)

    def set_unused_joint_goals_to_current(self):
        joint_goal = self.god_map.get_data([self._goal_identifier, str(Controller.JOINT)])
        for joint_name in self.controlled_joints:
            if joint_name not in joint_goal:
                joint_goal[joint_name] = {u'weight': 0.0,
                                          u'p_gain': 10,
                                          u'max_speed': self.get_robot().default_joint_velocity_limit,
                                          u'position': self.god_map.get_data([self._joint_states_identifier,
                                                                              joint_name,
                                                                              u'position'])}
                if joint_name not in self.used_joints:
                    joint_goal[joint_name][u'weight'] = 1

        self.god_map.set_data([self._goal_identifier, str(Controller.JOINT)], joint_goal)

    def get_expr_joint_current_position(self, joint_name):
        """
        :type joint_name: str
        :rtype: sw.Symbol
        """
        key = [self._joint_states_identifier, joint_name, u'position']
        return self.god_map.to_symbol(key)

    def get_expr_joint_goal_position(self, joint_name):
        """
        :type joint_name: str
        :rtype: sw.Symbol
        """
        key = [self._goal_identifier, str(Controller.JOINT), joint_name, u'position']
        return self.god_map.to_symbol(key)

    def get_expr_joint_goal_weight(self, joint_name):
        """
        :type joint_name: str
        :rtype: sw.Symbol
        """
        weight_key = [self._goal_identifier, str(Controller.JOINT), joint_name, u'weight']
        return self.god_map.to_symbol(weight_key)

    def get_expr_joint_goal_gain(self, joint_name):
        """
        :type joint_name: str
        :rtype: sw.Symbol
        """
        gain_key = [self._goal_identifier, str(Controller.JOINT), joint_name, u'p_gain']
        return self.god_map.to_symbol(gain_key)

    def get_expr_joint_goal_max_speed(self, joint_name):
        """
        :type joint_name: str
        :rtype: sw.Symbol
        """
        max_speed_key = [self._goal_identifier, str(Controller.JOINT), joint_name, u'max_speed']
        return self.god_map.to_symbol(max_speed_key)

    def get_expr_joint_distance_to_goal(self, joint_name):
        """
        :type joint_name: str
        :rtype: ShortestAngularDistanceInput
        """
        current_joint_key = [self._joint_states_identifier, joint_name, u'position']
        goal_joint_key = [self._goal_identifier, str(Controller.JOINT), joint_name, u'position']
        return ShortestAngularDistanceInput(self.god_map.to_symbol,
                                            [self._pyfunctions_identifier],
                                            current_joint_key,
                                            goal_joint_key)

    def add_js_controller_soft_constraints(self):
        """
        to self.controller and saves functions for continuous joints in god map
        """
        pyfunctions = {}
        for joint_name in self.controlled_joints:

            joint_current_expr = self.get_expr_joint_current_position(joint_name)
            goal_joint_expr = self.get_expr_joint_goal_position(joint_name)
            weight_expr = self.get_expr_joint_goal_weight(joint_name)
            gain_expr = self.get_expr_joint_goal_gain(joint_name)
            max_speed_expr = self.get_expr_joint_goal_max_speed(joint_name)

            if self.get_robot().is_joint_continuous(joint_name):
                change = self.get_expr_joint_distance_to_goal(joint_name)
                pyfunctions[change.get_key()] = change
                soft_constraints = continuous_joint_position(joint_current_expr,
                                                             change.get_expression(),
                                                             weight_expr,
                                                             gain_expr,
                                                             max_speed_expr, joint_name)
                self.controller.update_soft_constraints(soft_constraints, self.god_map.get_registered_symbols())
            else:
                soft_constraints = joint_position(joint_current_expr, goal_joint_expr, weight_expr,
                                                  gain_expr, max_speed_expr, joint_name)
            self.controller.update_soft_constraints(soft_constraints, self.god_map.get_registered_symbols())

        self.god_map.set_data([self._pyfunctions_identifier], pyfunctions)

    def add_collision_avoidance_soft_constraints(self):
        soft_constraints = {}
        for link in list(self.controllable_links):
            point_on_link_input = Point3Input(self.god_map.to_symbol,
                                              prefix=[self._closest_point_identifier, link, u'position_on_a'])
            other_point_input = Point3Input(self.god_map.to_symbol,
                                            prefix=[self._closest_point_identifier, link, u'position_on_b'])
            current_input = FrameInput(self.god_map.to_symbol,
                                       translation_prefix=[self._fk_identifier,
                                                           (self.root, link),
                                                           u'pose',
                                                           u'position'],
                                       rotation_prefix=[self._fk_identifier,
                                                        (self.root, link),
                                                        u'pose',
                                                        u'orientation'])
            min_dist = self.god_map.to_symbol([self._closest_point_identifier, link, u'min_dist'])
            contact_normal = Vector3Input(self.god_map.to_symbol,
                                          prefix=[self._closest_point_identifier, link, u'contact_normal'])

            soft_constraints.update(link_to_link_avoidance(link,
                                                           self.get_robot().get_fk_expression(self.root, link),
                                                           current_input.get_frame(),
                                                           point_on_link_input.get_expression(),
                                                           other_point_input.get_expression(),
                                                           contact_normal.get_expression(),
                                                           min_dist))

        self.controller.update_soft_constraints(soft_constraints, self.god_map.get_registered_symbols())

    def add_cart_controller_soft_constraints(self):
        print(u'used chains:')
        for t in [Controller.TRANSLATION_3D, Controller.ROTATION_3D]:
            for (root, tip), value in self.god_map.get_data([self._goal_identifier, str(t)]).items():
                self.used_joints.update(self.get_robot().get_joint_names_from_chain_controllable(root, tip))
                print(u'{} -> {} type: {}'.format(root, tip, t))
                self.controller.update_soft_constraints(self.cart_goal_to_soft_constraints(root, tip, t),
                                                        self.god_map.get_registered_symbols())

    def cart_goal_to_soft_constraints(self, root, tip, type):
        """
        :type root: str
        :type tip: str
        :param type: as defined in Controller msg
        :type type: int
        :rtype: dict
        """
        # TODO split this into 2 functions, for translation and rotation
        goal_input = FrameInput(self.god_map.to_symbol,
                                translation_prefix=[self._goal_identifier,
                                                    str(Controller.TRANSLATION_3D),
                                                    (root, tip),
                                                    u'goal_pose',
                                                    u'pose',
                                                    u'position'],
                                rotation_prefix=[self._goal_identifier,
                                                 str(Controller.ROTATION_3D),
                                                 (root, tip),
                                                 u'goal_pose',
                                                 u'pose',
                                                 u'orientation'])

        current_input = FrameInput(self.god_map.to_symbol,
                                   translation_prefix=[self._fk_identifier,
                                                       (root, tip),
                                                       u'pose',
                                                       u'position'],
                                   rotation_prefix=[self._fk_identifier,
                                                    (root, tip),
                                                    u'pose',
                                                    u'orientation'])
        weight_key = [self._goal_identifier, str(type), (root, tip), u'weight']
        weight = self.god_map.to_symbol(weight_key)
        p_gain_key = [self._goal_identifier, str(type), (root, tip), u'p_gain']
        p_gain = self.god_map.to_symbol(p_gain_key)
        max_speed_key = [self._goal_identifier, str(type), (root, tip), u'max_speed']
        max_speed = self.god_map.to_symbol(max_speed_key)

        if type == Controller.TRANSLATION_3D:
            return position_conv(goal_input.get_position(),
                                 sw.position_of(self.get_robot().get_fk_expression(root, tip)),
                                 weights=weight,
                                 trans_gain=p_gain,
                                 max_trans_speed=max_speed,
                                 ns=u'{}/{}'.format(root, tip))
        elif type == Controller.ROTATION_3D:
            return rotation_conv(goal_input.get_rotation(),
                                 sw.rotation_of(self.get_robot().get_fk_expression(root, tip)),
                                 current_input.get_rotation(),
                                 weights=weight,
                                 rot_gain=p_gain,
                                 max_rot_speed=max_speed,
                                 ns=u'{}/{}'.format(root, tip))

        return {}
