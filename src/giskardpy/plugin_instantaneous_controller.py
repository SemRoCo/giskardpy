import inspect
import itertools
import json
import traceback
from copy import copy
from multiprocessing import Process
from time import time, sleep

from giskard_msgs.msg import MoveGoal, MoveCmd
from py_trees import Status
from rospy_message_converter.message_converter import convert_ros_message_to_dictionary

import giskardpy.constraints
from giskardpy.constraints import LinkToAnyAvoidance, JointPosition
from giskardpy.exceptions import InsolvableException, ImplementationException
import giskardpy.identifier as identifier
from giskardpy.plugin import GiskardBehavior
from giskardpy.plugin_action_server import GetGoal
from giskardpy.symengine_controller import SymEngineController
from giskardpy.tfwrapper import transform_pose
from giskardpy import logging

def allowed_constraint_names():
    return [x[0] for x in inspect.getmembers(giskardpy.constraints) if inspect.isclass(x[1])]

# TODO split constraints into separate files
# TODO plan only not supported
# TODO waypoints not supported
class GoalToConstraints(GetGoal):
    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)
        self.used_joints = set()

        self.known_constraints = set()
        self.controlled_joints = set()
        self.controllable_links = set()
        self.last_urdf = None
        self.zero_weight_distance = self.get_god_map().safe_get_data(identifier.zero_weight_distance)
        self.low_weight_distance = self.get_god_map().safe_get_data(identifier.low_weight_distance)
        self.max_weight_distance = self.get_god_map().safe_get_data(identifier.max_weight_distance)

    def initialise(self):
        self.get_god_map().safe_set_data(identifier.collision_goal_identifier, None)

    def terminate(self, new_status):
        super(GoalToConstraints, self).terminate(new_status)

    def update(self):
        # TODO make this interruptable
        # TODO try catch everything

        goal_msg = self.get_goal()  # type: MoveGoal
        if len(goal_msg.cmd_seq) == 0:
            self.raise_to_blackboard(InsolvableException(u'goal empty'))
            return Status.SUCCESS
        if goal_msg.type not in [MoveGoal.PLAN_AND_EXECUTE, MoveGoal.PLAN_ONLY]:
            self.raise_to_blackboard(InsolvableException(u'invalid move action goal type: {}'.format(goal_msg.type)))
            return Status.SUCCESS
        self.get_god_map().safe_set_data(identifier.execute, goal_msg.type==MoveGoal.PLAN_AND_EXECUTE)

        self.get_god_map().safe_set_data(identifier.constraints_identifier, {})

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

        self.get_god_map().safe_set_data(identifier.collision_goal_identifier, move_cmd.collisions)

        self.get_god_map().safe_set_data(identifier.soft_constraint_identifier, self.soft_constraints)
        self.get_blackboard().runtime = time()
        return Status.SUCCESS

    def parse_constraints(self, cmd):
        """
        :type cmd: MoveCmd
        :rtype: dict
        """
        for constraint in itertools.chain(cmd.constraints, cmd.joint_constraints, cmd.cartesian_constraints):
            if constraint.type not in allowed_constraint_names():
                # TODO test me
                raise InsolvableException(u'unknown constraint')
            try:
                C = eval(u'giskardpy.constraints.{}'.format(constraint.type))
            except NameError as e:
                # TODO return next best constraint type
                raise ImplementationException(u'unsupported constraint type')
            try:
                if hasattr(constraint, u'parameter_value_pair'):
                    params = json.loads(constraint.parameter_value_pair)
                else:
                    params = convert_ros_message_to_dictionary(constraint)
                    del params[u'type']
                c = C(self.god_map, **params)
                soft_constraints = c.get_constraint()
                self.soft_constraints.update(soft_constraints)
            except TypeError as e:
                traceback.print_exc()
                raise ImplementationException(help(c.get_constraint))

    def add_js_controller_soft_constraints(self):
        for joint_name in self.get_robot().controlled_joints:
            c = JointPosition(self.get_god_map(), joint_name, self.get_robot().joint_state[joint_name].position, 0, 0)
            self.soft_constraints.update(c.get_constraint())

    def has_robot_changed(self):
        new_urdf = self.get_robot().get_urdf_str()
        result = self.last_urdf != new_urdf
        self.last_urdf = new_urdf
        return result

    def add_collision_avoidance_soft_constraints(self):
        """
        Adds a constraint for each link that pushed it away from its closest point.
        """
        soft_constraints = {}
        for link in self.get_robot().get_controlled_links():
            constraint = LinkToAnyAvoidance(self.god_map, link,
                                            max_weight_distance=self.get_god_map().safe_get_data(identifier.collisions+
                                                                                                 [link, u'max_weight_distance']),
                                            low_weight_distance=self.get_god_map().safe_get_data(identifier.collisions+
                                                                                                 [link, u'low_weight_distance']),
                                            zero_weight_distance=self.get_god_map().safe_get_data(identifier.collisions+
                                                                                                 [link, u'zero_weight_distance']))
            soft_constraints.update(constraint.get_constraint())

        self.soft_constraints.update(soft_constraints)


class ControllerPlugin(GiskardBehavior):
    def __init__(self, name):
        super(ControllerPlugin, self).__init__(name)
        self.path_to_functions = self.get_god_map().safe_get_data(identifier.data_folder)
        self.nWSR = self.get_god_map().safe_get_data(identifier.nWSR)
        self.soft_constraints = None

    def initialise(self):
        super(ControllerPlugin, self).initialise()
        self.init_controller()
        self.next_cmd = {}

    def setup(self, timeout=0.0):
        return super(ControllerPlugin, self).setup(5.0)

    def init_controller(self):
        new_soft_constraints = self.get_god_map().safe_get_data(identifier.soft_constraint_identifier)
        if self.soft_constraints is None or set(self.soft_constraints.keys()) != set(new_soft_constraints.keys()):
            self.soft_constraints = copy(new_soft_constraints)
            self.controller = SymEngineController(self.get_robot(),
                                                  u'{}/{}/'.format(self.path_to_functions, self.get_robot().get_name()))
            self.controller.set_controlled_joints(self.get_robot().controlled_joints)
            self.controller.update_soft_constraints(self.soft_constraints)
            p = Process(target=self.controller.compile)
            p.start()
            while p.is_alive():
                sleep(0.05)
            p.join()
            self.controller.compile()

    
    def update(self):
        expr = self.controller.get_expr()
        expr = self.god_map.get_symbol_map(expr)
        next_cmd = self.controller.get_cmd(expr, self.nWSR)
        self.next_cmd.update(next_cmd)

        self.get_god_map().safe_set_data(identifier.next_cmd_identifier, self.next_cmd)
        return Status.RUNNING
