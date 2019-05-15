import inspect
import json
from copy import copy
from time import time

from giskard_msgs.msg import MoveGoal, MoveCmd
from py_trees import Status

import giskardpy.constraints
from giskardpy.constraints import LinkToAnyAvoidance, JointPosition
from giskardpy.exceptions import InsolvableException, ImplementationException
from giskardpy.identifier import soft_constraint_identifier, next_cmd_identifier, \
    collision_goal_identifier, constraints_identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.plugin_action_server import GetGoal
from giskardpy.symengine_controller import SymEngineController
from giskardpy.tfwrapper import transform_pose
from giskardpy import logging

def allowed_constraint_names():
    return [x[0] for x in inspect.getmembers(giskardpy.constraints) if inspect.isclass(x[1])]


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

        self.get_god_map().safe_set_data(constraints_identifier, {})

        if self.has_robot_changed():
            self.soft_constraints = {}
            # TODO split soft contraints into js, coll and cart; update cart always and js/coll only when urdf changed, js maybe never
            self.add_js_controller_soft_constraints()
        self.add_collision_avoidance_soft_constraints()

        # TODO handle multiple cmds
        move_cmd = goal_msg.cmd_seq[0]  # type: MoveCmd
        try:
            self.parse_constraints(move_cmd)
        except AttributeError:
            self.raise_to_blackboard(InsolvableException(u'couldn\'t transform goal'))
            return Status.SUCCESS
        except InsolvableException as e:
            self.raise_to_blackboard(e)
            return Status.SUCCESS

        # self.set_unused_joint_goals_to_current()

        self.get_god_map().safe_set_data(collision_goal_identifier, move_cmd.collisions)

        self.get_god_map().safe_set_data(soft_constraint_identifier, self.soft_constraints)
        self.get_blackboard().runtime = time()
        return Status.SUCCESS

    def parse_constraints(self, cmd):
        """
        :type cmd: MoveCmd
        :rtype: dict
        """
        for constraint in cmd.constraints:
            if constraint.name not in allowed_constraint_names():
                # TODO test me
                raise InsolvableException(u'unknown constraint')
            try:
                C = eval(u'giskardpy.constraints.{}'.format(constraint.name))
            except NameError as e:
                # TODO return next best constraint type
                raise ImplementationException(u'unsupported constraint type')
            try:
                params = json.loads(constraint.parameter_value_pair)
                c = C(self.god_map, **params)
                soft_constraints = c.get_constraint()
                self.soft_constraints.update(soft_constraints)
            except TypeError as e:
                raise ImplementationException(help(c.get_constraint))

    def add_js_controller_soft_constraints(self):
        for joint_name in self.get_robot().controlled_joints:
            c = JointPosition(self.get_god_map(), joint_name, self.get_robot().joint_state[joint_name].position, 0, 0)
            self.soft_constraints.update(c.get_constraint())

    def has_robot_changed(self):
        new_urdf = self.get_robot().get_urdf()
        result = self.last_urdf != new_urdf
        self.last_urdf = new_urdf
        return result

    def add_collision_avoidance_soft_constraints(self):
        """
        Adds a constraint for each link that pushed it away from its closest point.
        """
        soft_constraints = {}
        for link in self.get_robot().get_controlled_links():
            constraint = LinkToAnyAvoidance(self.god_map, link)
            soft_constraints.update(constraint.get_constraint())

        self.soft_constraints.update(soft_constraints)


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
