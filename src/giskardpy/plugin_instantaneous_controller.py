from copy import copy
import rospy
from giskardpy.input_system import JointStatesInput, FrameInput, Point3Input
from giskardpy.plugin import Plugin
from giskardpy.symengine_controller import JointController, Controller, position_conv, rotation_conv, \
    link_to_any_avoidance
import symengine_wrappers as sw


class ControllerPlugin(Plugin):
    def __init__(self, js_identifier='js', goal_identifier='goal', next_cmd_identifier='motor'):
        self._joint_states_identifier = js_identifier
        self._goal_identifier = goal_identifier
        self._next_cmd_identifier = next_cmd_identifier
        self._controller = None
        super(ControllerPlugin, self).__init__()

    def get_readings(self):
        if len(self.next_cmd) > 0:
            updates = {self._next_cmd_identifier: self.next_cmd}
            return updates
        return {}

    def update(self):
        if self.god_map.get_data(self._goal_identifier) is not None:
            next_cmd = self._controller.get_cmd(self.god_map.get_expr_values())
            self.next_cmd.update(next_cmd)

    def start(self, god_map):
        self.next_cmd = {}
        super(ControllerPlugin, self).start(god_map)

    def stop(self):
        pass

    def get_replacement_parallel_universe(self):
        return copy(self)

    def __copy__(self):
        cp = self.__class__()
        cp._controller = self._controller
        return cp


class JointControllerPlugin(ControllerPlugin):
    def __init__(self, js_identifier='js', goal_identifier='joint_goal', next_cmd_identifier='motor'):
        super(JointControllerPlugin, self).__init__(js_identifier=js_identifier,
                                                    goal_identifier=goal_identifier,
                                                    next_cmd_identifier=next_cmd_identifier)

    def start(self, god_map):
        super(JointControllerPlugin, self).start(god_map)
        if self._controller is None:
            urdf = rospy.get_param('robot_description')
            self._controller = JointController(urdf)
            current_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                                 self._controller.robot.get_joint_names(),
                                                                 self._joint_states_identifier,
                                                                 'position')
            goal_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                              self._controller.robot.get_joint_names(),
                                                              self._goal_identifier,
                                                              'position')
            self._controller.init(current_joints, goal_joints)


class CartesianControllerPlugin(ControllerPlugin):
    def __init__(self, roots, tips, js_identifier='js', fk_identifier='fk', goal_identifier='cartesian_goal',
                 next_cmd_identifier='motor'):
        self.fk_identifier = fk_identifier
        self.roots = roots
        self.tips = tips
        super(CartesianControllerPlugin, self).__init__(js_identifier=js_identifier,
                                                        goal_identifier=goal_identifier,
                                                        next_cmd_identifier=next_cmd_identifier)

    def start(self, god_map):
        super(CartesianControllerPlugin, self).start(god_map)
        if self._controller is None:
            urdf = rospy.get_param('robot_description')
            self._controller = Controller(urdf)
            robot = self._controller.robot
            current_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                                 robot.get_joint_names(),
                                                                 self._joint_states_identifier,
                                                                 'position')

            robot.set_joint_symbol_map(current_joints)

            for root, tip in zip(self.roots, self.tips):
                trans_prefix = '{}/{},{}/translation'.format(self._goal_identifier, root, tip)
                rot_prefix = '{}/{},{}/rotation'.format(self._goal_identifier, root, tip)
                goal_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)

                trans_prefix = '{}/{},{}/pose/position'.format(self.fk_identifier, root, tip)
                rot_prefix = '{}/{},{}/pose/orientation'.format(self.fk_identifier, root, tip)
                current_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)

                self._controller.add_constraints(position_conv(goal_input.get_position(),
                                                               sw.pos_of(robot.get_fk_expression(root, tip)),
                                                               ns='{}/{}'.format(root, tip)))
                self._controller.add_constraints(rotation_conv(goal_input.get_rotation(),
                                                               sw.rot_of(robot.get_fk_expression(root, tip)),
                                                               current_input.get_rotation(),
                                                               ns='{}/{}'.format(root, tip)))

            joint_names = set()
            for root, tip in zip(self.roots, self.tips):
                joint_names.update(self._controller.robot.get_chain_joints(root, tip))
            self._controller.init(controlled_joints=joint_names)

    def __copy__(self):
        # TODO potential bug, should pass identifier
        cp = self.__class__(self.roots, self.tips)
        cp._controller = self._controller
        return cp


class CartesianBulletControllerPlugin(ControllerPlugin):
    def __init__(self, roots, tips, js_identifier='js', fk_identifier='fk', goal_identifier='cartesian_goal',
                 next_cmd_identifier='motor', collision_identifier='collision', closest_point_identifier='cpi'):
        self.fk_identifier = fk_identifier
        self.collision_identifier = collision_identifier
        self.closest_point_identifier = closest_point_identifier
        self.roots = roots
        self.tips = tips
        super(CartesianBulletControllerPlugin, self).__init__(js_identifier=js_identifier,
                                                              goal_identifier=goal_identifier,
                                                              next_cmd_identifier=next_cmd_identifier)

    def start(self, god_map):
        super(CartesianBulletControllerPlugin, self).start(god_map)
        if self._controller is None:
            urdf = rospy.get_param('robot_description')
            self._controller = Controller(urdf)
            robot = self._controller.robot

            current_joints = JointStatesInput.prefix_constructor(self.god_map.get_expr,
                                                                 robot.get_joint_names(),
                                                                 self._joint_states_identifier,
                                                                 'position')
            robot.set_joint_symbol_map(current_joints)

            added_links = set()
            for root, tip in zip(self.roots, self.tips):

                trans_prefix = '{}/{},{}/translation'.format(self._goal_identifier, root, tip)
                rot_prefix = '{}/{},{}/rotation'.format(self._goal_identifier, root, tip)
                goal_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)

                trans_prefix = '{}/{},{}/pose/position'.format(self.fk_identifier, root, tip)
                rot_prefix = '{}/{},{}/pose/orientation'.format(self.fk_identifier, root, tip)
                current_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)

                self._controller.add_constraints(position_conv(goal_input.get_position(),
                                                               sw.pos_of(robot.get_fk_expression(root, tip)),
                                                               ns='{}/{}'.format(root, tip)))
                self._controller.add_constraints(rotation_conv(goal_input.get_rotation(),
                                                               sw.rot_of(robot.get_fk_expression(root, tip)),
                                                               current_input.get_rotation(),
                                                               ns='{}/{}'.format(root, tip)))

                for link1 in robot.get_link_tree(root, tip):
                    if link1 not in added_links:
                        added_links.add(link1)
                        point_on_link_input = Point3Input.position_on_a_constructor(self.god_map.get_expr,
                                                                                    '{}/{}'.format(
                                                                                        self.closest_point_identifier,
                                                                                        link1))
                        other_point_input = Point3Input.position_on_b_constructor(self.god_map.get_expr,
                                                                                  '{}/{}'.format(
                                                                                      self.closest_point_identifier,
                                                                                      link1))
                        trans_prefix = '{}/{},{}/pose/position'.format(self.fk_identifier, root, link1)
                        rot_prefix = '{}/{},{}/pose/orientation'.format(self.fk_identifier, root, link1)
                        current_input = FrameInput.prefix_constructor(trans_prefix, rot_prefix, self.god_map.get_expr)
                        self._controller.add_constraints(link_to_any_avoidance(link1,
                                                                               robot.get_fk_expression(root, link1),
                                                                               current_input.get_frame(),
                                                                               point_on_link_input.get_expression(),
                                                                               other_point_input.get_expression()))

            joint_names = set()
            for root, tip in zip(self.roots, self.tips):
                joint_names.update(self._controller.robot.get_chain_joints(root, tip))
            self._controller.init(controlled_joints=joint_names, free_symbols=self.god_map.get_free_symbols())

    def __copy__(self):
        cp = self.__class__(self.roots, self.tips)
        cp._controller = self._controller
        return cp
