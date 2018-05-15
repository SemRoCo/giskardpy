from collections import OrderedDict, defaultdict
from copy import copy
import rospy
from giskard_msgs.msg import Controller

from giskardpy.input_system import JointStatesInput, FrameInput, Point3Input
from giskardpy.plugin import Plugin
from giskardpy.symengine_controller import SymEngineController, position_conv, rotation_conv, \
    link_to_any_avoidance, joint_position
import symengine_wrappers as sw
from giskardpy.utils import keydefaultdict


class CartesianBulletControllerPlugin(Plugin):
    def __init__(self, root, js_identifier, fk_identifier, goal_identifier, next_cmd_identifier,
                 collision_identifier, closest_point_identifier, controlled_joints_identifier):
        """
        :param roots:
        :type roots: list
        :param tips:
        :type tips: list
        :param js_identifier:
        :param fk_identifier:
        :param goal_identifier:
        :param next_cmd_identifier:
        :param collision_identifier:
        :param closest_point_identifier:
        """
        self.controlled_joints_identifier = controlled_joints_identifier
        self._fk_identifier = fk_identifier
        self._collision_identifier = collision_identifier
        self._closest_point_identifier = closest_point_identifier
        self.root = root
        self.soft_constraints = OrderedDict()
        self._joint_states_identifier = js_identifier
        self._goal_identifier = goal_identifier
        self._next_cmd_identifier = next_cmd_identifier
        self._controller = None
        self.known_constraints = set()
        self.controlled_joints = set()
        self.controller_updated = False
        super(CartesianBulletControllerPlugin, self).__init__()

    def update(self):
        if self.god_map.get_data(self._goal_identifier) is not None:
            # TODO set joint goals

            # update controlled joints
            new_controlled_joints = self.god_map.get_data(self.controlled_joints_identifier)
            if len(set(new_controlled_joints).difference(self.controlled_joints)) != 0:
                self._controller.set_controlled_joints(new_controlled_joints)
            self.controlled_joints = new_controlled_joints

            if not self.controller_updated:
                self.modify_controller()

            next_cmd = self._controller.get_cmd(self.god_map.get_expr_values())
            self.next_cmd.update(next_cmd)

    def get_readings(self):
        if len(self.next_cmd) > 0:
            updates = {self._next_cmd_identifier: self.next_cmd}
            return updates
        return {}

    def start_always(self):
        self.next_cmd = {}

    def stop(self):
        pass

    def start_once(self):
        urdf = rospy.get_param('robot_description')
        self._controller = SymEngineController(urdf)
        robot = self._controller.robot

        current_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                             robot.get_joint_names(),
                                                             self._joint_states_identifier,
                                                             'position')
        robot.set_joint_symbol_map(current_joints)

    def modify_controller(self):
        rebuild_controller = False
        robot = self._controller.robot
        hold_joints = set(self.controlled_joints)
        if not self._controller.is_initialized():
            rebuild_controller = True
            controllable_links = set()
            for joint_name in self.controlled_joints:
                current_joint_key = self.god_map.get_expr([self._joint_states_identifier, joint_name, 'position'])
                goal_joint_key = self.god_map.get_expr(
                    [self._goal_identifier, Controller.JOINT, joint_name, 'position'])
                weight = self.god_map.get_expr([self._goal_identifier, Controller.JOINT, joint_name, 'weight'])
                self.soft_constraints[joint_name] = joint_position(current_joint_key, goal_joint_key, weight)
                controllable_links.update(robot.get_link_tree(joint_name))

            for link in list(controllable_links):
                point_on_link_input = Point3Input.position_on_a_constructor(self.god_map.get_expr,
                                                                            '{}/{}'.format(
                                                                                self._closest_point_identifier,
                                                                                link))
                other_point_input = Point3Input.position_on_b_constructor(self.god_map.get_expr,
                                                                          '{}/{}'.format(
                                                                              self._closest_point_identifier,
                                                                              link))
                trans_prefix = '{}/{},{}/pose/position'.format(self._fk_identifier, self.root, link)
                rot_prefix = '{}/{},{}/pose/orientation'.format(self._fk_identifier, self.root, link)
                current_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)
                self.soft_constraints.update(link_to_any_avoidance(link,
                                                                   robot.get_fk_expression(self.root, link),
                                                                   current_input.get_frame(),
                                                                   point_on_link_input.get_expression(),
                                                                   other_point_input.get_expression()))

        for (root, tip), value in self.god_map.get_data([self._goal_identifier, Controller.TRANSLATION_3D]).items():
            key = '{}/{},{}'.format(Controller.TRANSLATION_3D, root, tip)
            hold_joints.difference_update(robot.get_chain_joints(root, tip))
            if key not in self.known_constraints:
                print('added chain root: {} tip: {} type: TRANSLATION_3D'.format(root, tip))
                self.known_constraints.add(key)
                self.soft_constraints.update(self.controller_msg_to_constraint(root, tip, Controller.TRANSLATION_3D))
                rebuild_controller = True
            self.god_map.set_data([self._goal_identifier, Controller.TRANSLATION_3D, ','.join([root, tip]), 'weight'], 1)
        for (root, tip), value in self.god_map.get_data([self._goal_identifier, Controller.ROTATION_3D]).items():
            key = '{}/{},{}'.format(Controller.ROTATION_3D, root, tip)
            hold_joints.difference_update(robot.get_chain_joints(root, tip))
            if key not in self.known_constraints:
                print('added chain root: {} tip: {} type: ROTATION_3D'.format(root, tip))
                self.known_constraints.add(key)
                self.soft_constraints.update(self.controller_msg_to_constraint(root, tip, Controller.ROTATION_3D))
                rebuild_controller = True
            self.god_map.set_data([self._goal_identifier, Controller.ROTATION_3D, ','.join([root, tip]), 'weight'], 1)

        # TODO handle joint controller

        # set weight of unused joints to 0
        # TODO this breaks the rule of not modifying the god map inside of a plugin
        joint_goal = self.god_map.get_data([self._goal_identifier, Controller.JOINT])
        for joint_name in self.controlled_joints:
            if joint_name not in joint_goal:
                joint_goal[joint_name] = {'weight': 0,
                                          'position': self.god_map.get_data([self._joint_states_identifier,
                                                                             joint_name,
                                                                             'position'])}
                if joint_name in hold_joints:
                    joint_goal[joint_name]['weight'] = 1

        self.god_map.set_data([self._goal_identifier, Controller.JOINT], joint_goal)

        if rebuild_controller or self._controller is None:
            # TODO prevent this from being called too often
            # TODO turn off old constraints by setting weight to 0
            # TODO sanity checking

            self._controller.init(self.soft_constraints, self.god_map.get_free_symbols())
        self.controller_updated = True

    def controller_msg_to_constraint(self, root, tip, type):
        """
        :param controller_msg:
        :type controller_msg: Controller
        :return:
        :rtype: dict
        """
        robot = self._controller.robot

        trans_prefix = '{}/{}/{},{}/goal_pose/pose/position'.format(self._goal_identifier, Controller.TRANSLATION_3D,
                                                                    root, tip)
        rot_prefix = '{}/{}/{},{}/goal_pose/pose/orientation'.format(self._goal_identifier, Controller.ROTATION_3D,
                                                                     root, tip)
        goal_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)

        trans_prefix = '{}/{},{}/pose/position'.format(self._fk_identifier, root, tip)
        rot_prefix = '{}/{},{}/pose/orientation'.format(self._fk_identifier, root, tip)
        current_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)
        weight = self.god_map.get_expr([self._goal_identifier, str(type), ','.join([root, tip]), 'weight'])
        p_gain = self.god_map.get_expr([self._goal_identifier, str(type), ','.join([root, tip]), 'p_gain'])
        max_speed = self.god_map.get_expr([self._goal_identifier, str(type), ','.join([root, tip]), 'threshold_value'])

        if type == Controller.TRANSLATION_3D:
            return position_conv(goal_input.get_position(),
                                 sw.pos_of(robot.get_fk_expression(root, tip)),
                                 weights=weight,
                                 trans_gain=p_gain,
                                 max_trans_speed=max_speed,
                                 ns='{}/{}'.format(root, tip))
        elif type == Controller.ROTATION_3D:
            return rotation_conv(goal_input.get_rotation(),
                                 sw.rot_of(robot.get_fk_expression(root, tip)),
                                 current_input.get_rotation(),
                                 weights=weight,
                                 rot_gain=p_gain,
                                 max_rot_speed=max_speed,
                                 ns='{}/{}'.format(root, tip))

        return {}

    def copy(self):
        cp = self.__class__(self.root, self._joint_states_identifier, self._fk_identifier,
                            self._goal_identifier, self._next_cmd_identifier, self._collision_identifier,
                            self._closest_point_identifier, self.controlled_joints_identifier)
        # TODO not cool that you always have to copy the controller here
        cp._controller = self._controller
        cp.soft_constraints = self.soft_constraints
        cp.known_constraints = self.known_constraints
        # cp.controlled_joints = self.controlled_joints
        return cp
