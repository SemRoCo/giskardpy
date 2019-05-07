import json
import numpy as np
from copy import copy
from time import time

from giskard_msgs.msg import Controller, MoveGoal, MoveCmd, Constraint
from py_trees import Status

import symengine_wrappers as sw
from giskardpy.constraints import JointPosition
from giskardpy.exceptions import InsolvableException
from giskardpy.identifier import soft_constraint_identifier, next_cmd_identifier, \
    collision_goal_identifier, fk_identifier, \
    closest_point_identifier, js_identifier, default_joint_vel_identifier, constraints_identifier
from giskardpy.input_system import FrameInput, Point3Input, Vector3Input
from giskardpy.plugin import GiskardBehavior
from giskardpy.plugin_action_server import GetGoal
from giskardpy.symengine_controller import SymEngineController, position_conv, rotation_conv, \
    link_to_link_avoidance, joint_position, continuous_joint_position, rotation_conv_slerp
from giskardpy.tfwrapper import transform_pose


# TODO plan only not supported
# TODO waypoints not supported
class GoalToConstraints(GetGoal):
    def __init__(self, name, as_name, use_slerp=True):
        GetGoal.__init__(self, name, as_name)
        self.used_joints = set()

        self.known_constraints = set()
        self.controlled_joints = set()
        self.controllable_links = set()
        self.last_urdf = None
        self.use_slerp = use_slerp

    def initialise(self):
        self.get_god_map().safe_set_data(collision_goal_identifier, None)

    def terminate(self, new_status):
        super(GoalToConstraints, self).terminate(new_status)

    def update(self):
        # TODO make this interruptable

        goal_msg = self.get_goal()  # type: MoveGoal
        if len(goal_msg.cmd_seq) == 0:
            self.raise_to_blackboard(InsolvableException(u'goal empty'))
            return Status.SUCCESS
        if goal_msg.type != MoveGoal.PLAN_AND_EXECUTE:
            self.raise_to_blackboard(InsolvableException(u'only plan and execute is supported'))
            return Status.SUCCESS

        if self.has_robot_changed():
            self.soft_constraints = {}
            # TODO split soft contraints into js, coll and cart; update cart always and js/coll only when urdf changed, js maybe never
            # self.add_js_controller_soft_constraints()
            # self.add_collision_avoidance_soft_constraints()

        # TODO handle multiple cmds
        move_cmd = goal_msg.cmd_seq[0]  # type: MoveCmd
        # for constraint in move_cmd.constraints:  # type: Constraint
        #     try:
        #         c = eval(constraint.name)(self.god_map, u'constraints')
        #     except NameError as e:
        #         self.raise_to_blackboard(InsolvableException(u'unsupported controller type'))
        #         return Status.SUCCESS
        #     try:
        #         soft_constraints = c.get_constraint(**json.loads(constraint.parameter_value_pair))
        #     # if controller.type in [Controller.TRANSLATION_3D, Controller.ROTATION_3D]:
        #     #     self.add_cart_controller_soft_constraints(controller, controller.type)
        #         # god_map_rdy_goal = cart_controller_to_goal(controller)
        #     # elif controller.type in [Controller.JOINT]:
        #     #     pass
        #     except TypeError as e:
        #         self.raise_to_blackboard(InsolvableException(help(c)))
        try:
            parsed_constraints = self.parse_constraints(move_cmd)
        except AttributeError:
            self.raise_to_blackboard(InsolvableException(u'couldn\'t transform goal'))
            return Status.SUCCESS
        self.get_god_map().safe_set_data(constraints_identifier, parsed_constraints)

        # self.set_unused_joint_goals_to_current()

        self.get_god_map().safe_set_data(collision_goal_identifier, [])

        self.get_god_map().safe_set_data(soft_constraint_identifier, self.soft_constraints)
        self.get_blackboard().runtime = time()
        return Status.SUCCESS

    def parse_constraints(self, cmd):
        """
        :type cmd: MoveCmd
        :rtype: dict
        """
        constraints = {}
        for constraint in cmd.constraints:
            try:
                c = eval(constraint.name)(self.god_map, constraints_identifier)
            except NameError as e:
                # TODO return next best constraint type
                self.raise_to_blackboard(InsolvableException(u'unsupported constraint type'))
                return Status.SUCCESS
            try:
                params = json.loads(constraint.parameter_value_pair)
                soft_constraints = c.get_constraint(**params)
                self.soft_constraints.update(soft_constraints)
                constraints[str(c)] = params
            except TypeError as e:
                self.raise_to_blackboard(InsolvableException(help(c.get_constraint)))
        self.get_god_map().safe_set_data(constraints_identifier, constraints)

    def has_robot_changed(self):
        new_urdf = self.get_robot().get_urdf()
        result = self.last_urdf != new_urdf
        self.last_urdf = new_urdf
        return result

    # def add_js_controller_soft_constraints(self):
    #     """
    #     to self.controller and saves functions for continuous joints in god map.
    #     """
    #     for joint_name in self.get_robot().controlled_joints:
    #         c = JointPosition(self.god_map, [u'constraints'])
    #         self.soft_constraints.update(c.get_constraint(joint_name))

    # def add_cart_controller_soft_constraints(self, controller, t):
    #     """
    #     Adds cart controller constraints for each goal.
    #     :type controller: Controller
    #     """
    #     print(u'used chains:')
    #     (root, tip) = (controller.root_link, controller.tip_link)
    #     self.used_joints.update(self.get_robot().get_joint_names_from_chain_controllable(root, tip))
    #     print(u'{} -> {} type: {}'.format(root, tip, t))
    #     self.soft_constraints.update(self.cart_goal_to_soft_constraints(root, tip, t))

    # def cart_goal_to_soft_constraints(self, root, tip, type):
    #     """
    #     :type root: str
    #     :type tip: str
    #     :param type: as defined in Controller msg
    #     :type type: int
    #     :rtype: dict
    #     """
    #     # TODO split this into 2 functions, for translation and rotation
    #     goal_input = FrameInput(self.god_map.to_symbol,
    #                             translation_prefix=cartesian_goal_identifier +
    #                                                [str(Controller.TRANSLATION_3D),
    #                                                 (root, tip),
    #                                                 u'goal_pose',
    #                                                 u'pose',
    #                                                 u'position'],
    #                             rotation_prefix=cartesian_goal_identifier +
    #                                             [str(Controller.ROTATION_3D),
    #                                              (root, tip),
    #                                              u'goal_pose',
    #                                              u'pose',
    #                                              u'orientation'])
    #
    #     current_input = FrameInput(self.god_map.to_symbol,
    #                                translation_prefix=fk_identifier +
    #                                                   [(root, tip),
    #                                                    u'pose',
    #                                                    u'position'],
    #                                rotation_prefix=fk_identifier +
    #                                                [(root, tip),
    #                                                 u'pose',
    #                                                 u'orientation'])
    #     weight_key = cartesian_goal_identifier + [str(type), (root, tip), u'weight']
    #     weight = self.god_map.to_symbol(weight_key)
    #     p_gain_key = cartesian_goal_identifier + [str(type), (root, tip), u'p_gain']
    #     p_gain = self.god_map.to_symbol(p_gain_key)
    #     max_speed_key = cartesian_goal_identifier + [str(type), (root, tip), u'max_speed']
    #     max_speed = self.god_map.to_symbol(max_speed_key)
    #
    #     if type == Controller.TRANSLATION_3D:
    #         return position_conv(goal_input.get_position(),
    #                              sw.position_of(self.get_robot().get_fk_expression(root, tip)),
    #                              weights=weight,
    #                              trans_gain=p_gain,
    #                              max_trans_speed=max_speed,
    #                              ns=u'{}/{}'.format(root, tip))
    #     elif type == Controller.ROTATION_3D:
    #         if self.use_slerp:
    #             return rotation_conv_slerp(goal_input.get_rotation(),
    #                                        sw.rotation_of(self.get_robot().get_fk_expression(root, tip)),
    #                                        current_input.get_rotation(),
    #                                        weights=weight,
    #                                        rot_gain=p_gain,
    #                                        max_rot_speed=max_speed,
    #                                        ns=u'{}/{}'.format(root, tip))
    #         else:
    #             return rotation_conv(goal_input.get_rotation(),
    #                                  sw.rotation_of(self.get_robot().get_fk_expression(root, tip)),
    #                                  current_input.get_rotation(),
    #                                  weights=weight,
    #                                  rot_gain=p_gain,
    #                                  max_rot_speed=max_speed,
    #                                  ns=u'{}/{}'.format(root, tip))
    #
    #     return {}

    # def add_collision_avoidance_soft_constraints(self):
    #     """
    #     Adds a constraint for each link that pushed it away from its closest point.
    #     """
    #     soft_constraints = {}
    #     for link in self.get_robot().get_controlled_links():
    #         point_on_link_input = Point3Input(self.god_map.to_symbol,
    #                                           prefix=closest_point_identifier + [link, u'position_on_a'])
    #         other_point_input = Point3Input(self.god_map.to_symbol,
    #                                         prefix=closest_point_identifier + [link, u'position_on_b'])
    #         current_input = FrameInput(self.god_map.to_symbol,
    #                                    translation_prefix=fk_identifier +
    #                                                       [(self.get_robot().get_root(), link),
    #                                                        u'pose',
    #                                                        u'position'],
    #                                    rotation_prefix=fk_identifier +
    #                                                    [(self.get_robot().get_root(), link),
    #                                                     u'pose',
    #                                                     u'orientation'])
    #         min_dist = self.god_map.to_symbol(closest_point_identifier + [link, u'min_dist'])
    #         contact_normal = Vector3Input(self.god_map.to_symbol,
    #                                       prefix=closest_point_identifier + [link, u'contact_normal'])
    #
    #         soft_constraints.update(link_to_link_avoidance(link,
    #                                                        self.get_robot().get_fk_expression(
    #                                                            self.get_robot().get_root(), link),
    #                                                        current_input.get_frame(),
    #                                                        point_on_link_input.get_expression(),
    #                                                        other_point_input.get_expression(),
    #                                                        contact_normal.get_expression(),
    #                                                        min_dist))
    #
    #     self.soft_constraints.update(soft_constraints)

    # def set_unused_joint_goals_to_current(self):
    #     """
    #     Sets the goal for all joints which are not used in another goal to their current position.
    #     """
    #     joint_goal = self.get_god_map().safe_get_data(cartesian_goal_identifier + [str(Controller.JOINT)])
    #     for joint_name in self.get_robot().controlled_joints:
    #         if joint_name not in joint_goal:
    #             joint_goal[joint_name] = {u'weight': 0,
    #                                       u'p_gain': 0,
    #                                       u'max_speed': self.get_god_map().safe_get_data(default_joint_vel_identifier),
    #                                       u'position': self.get_god_map().safe_get_data(js_identifier +
    #                                                                                     [joint_name,
    #                                                                                      u'position'])}
    #             if joint_name not in self.used_joints:
    #                 joint_goal[joint_name][u'weight'] = 1
    #                 joint_goal[joint_name][u'p_gain'] = 10.  # FIXME don't hardcode this
    #
    #     self.get_god_map().safe_set_data(cartesian_goal_identifier + [str(Controller.JOINT)], joint_goal)

    # def get_expr_joint_current_position(self, joint_name):
    #     """
    #     :type joint_name: str
    #     :rtype: sw.Symbol
    #     """
    #     key = js_identifier + [joint_name, u'position']
    #     return self.god_map.to_symbol(key)
    #
    # def get_expr_joint_goal_position(self, joint_name):
    #     """
    #     :type joint_name: str
    #     :rtype: sw.Symbol
    #     """
    #     key = cartesian_goal_identifier + [str(Controller.JOINT), joint_name, u'position']
    #     return self.god_map.to_symbol(key)
    #
    # def get_expr_joint_goal_weight(self, joint_name):
    #     """
    #     :type joint_name: str
    #     :rtype: sw.Symbol
    #     """
    #     weight_key = cartesian_goal_identifier + [str(Controller.JOINT), joint_name, u'weight']
    #     return self.god_map.to_symbol(weight_key)
    #
    # def get_expr_joint_goal_gain(self, joint_name):
    #     """
    #     :type joint_name: str
    #     :rtype: sw.Symbol
    #     """
    #     gain_key = cartesian_goal_identifier + [str(Controller.JOINT), joint_name, u'p_gain']
    #     return self.god_map.to_symbol(gain_key)
    #
    # def get_expr_joint_goal_max_speed(self, joint_name):
    #     """
    #     :type joint_name: str
    #     :rtype: sw.Symbol
    #     """
    #     max_speed_key = cartesian_goal_identifier + [str(Controller.JOINT), joint_name, u'max_speed']
    #     return self.god_map.to_symbol(max_speed_key)

# def joint_controller_to_goal(controller):
#     """
#     :type controller: Controller
#     :return: joint_name -> {controller parameter -> value}
#     :rtype: dict
#     """
#     # TODO check for unknown joint names?
#     goals = {}
#     for i, joint_name in enumerate(controller.goal_state.name):
#         goals[joint_name] = {u'weight': controller.weight,
#                              u'p_gain': controller.p_gain,
#                              u'max_speed': controller.max_speed,
#                              u'position': controller.goal_state.position[i]}
#     return goals


# def cart_controller_to_goal(controller):
#     """
#     :type controller: Controller
#     :return: (root_link, tip_link) -> {controller parameter -> value}
#     :rtype: dict
#     """
#     goals = {}
#     root = controller.root_link
#     tip = controller.tip_link
#     controller.goal_pose = transform_pose(root, controller.goal_pose)
#     # make sure rotation is normalized quaternion
#     # TODO make a function out of this
#     rotation = np.array([controller.goal_pose.pose.orientation.x,
#                          controller.goal_pose.pose.orientation.y,
#                          controller.goal_pose.pose.orientation.z,
#                          controller.goal_pose.pose.orientation.w])
#     normalized_rotation = rotation / np.linalg.norm(rotation)
#     controller.goal_pose.pose.orientation.x = normalized_rotation[0]
#     controller.goal_pose.pose.orientation.y = normalized_rotation[1]
#     controller.goal_pose.pose.orientation.z = normalized_rotation[2]
#     controller.goal_pose.pose.orientation.w = normalized_rotation[3]
#     goals[root, tip] = controller
#     return goals


class ControllerPlugin(GiskardBehavior):
    def __init__(self, name, path_to_functions, nWSR=None):
        super(ControllerPlugin, self).__init__(name)
        self.path_to_functions = path_to_functions
        self.nWSR = nWSR
        self.soft_constraints = None

    def initialise(self):
        super(ControllerPlugin, self).initialise()
        self.init_controller()
        self.next_cmd = {}

    def setup(self, timeout=0.0):
        return super(ControllerPlugin, self).setup(5.0)

    def init_controller(self):
        new_soft_constraints = self.get_god_map().safe_get_data(soft_constraint_identifier)
        if self.soft_constraints is None or set(self.soft_constraints.keys()) != set(new_soft_constraints.keys()):
            self.soft_constraints = copy(new_soft_constraints)
            self.controller = SymEngineController(self.get_robot(),
                                                  u'{}/{}/'.format(self.path_to_functions, self.get_robot().get_name()))
            self.controller.set_controlled_joints(self.get_robot().controlled_joints)
            self.controller.update_soft_constraints(self.soft_constraints)
            self.controller.compile()

    def update(self):
        expr = self.controller.get_expr()
        expr = self.god_map.get_symbol_map(expr)
        next_cmd = self.controller.get_cmd(expr, self.nWSR)
        self.next_cmd.update(next_cmd)

        self.get_god_map().safe_set_data(next_cmd_identifier, self.next_cmd)
        return Status.RUNNING
