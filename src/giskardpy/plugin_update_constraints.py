import inspect
import itertools
import json
import traceback
import difflib
from time import time

from giskard_msgs.msg import MoveCmd
from py_trees import Status
from rospy_message_converter.message_converter import convert_ros_message_to_dictionary

import giskardpy.constraints
import giskardpy.identifier as identifier
from giskardpy.constraints import JointPosition, SelfCollisionAvoidance, ExternalCollisionAvoidance
from giskardpy.exceptions import InsolvableException, ImplementationException
from giskardpy.plugin_action_server import GetGoal


class GoalToConstraints(GetGoal):
    # FIXME no error msg when constraint has missing parameter
    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)
        self.used_joints = set()

        self.controlled_joints = set()
        self.controllable_links = set()
        self.last_urdf = None
        self.allowed_constraint_types = {x[0]:x[1] for x in inspect.getmembers(giskardpy.constraints) if inspect.isclass(x[1])}

    def initialise(self):
        self.get_god_map().safe_set_data(identifier.collision_goal_identifier, None)

    def update(self):
        # TODO make this interruptable
        # TODO try catch everything

        move_cmd = self.get_god_map().safe_get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.FAILURE

        self.get_god_map().safe_set_data(identifier.constraints_identifier, {})

        self.soft_constraints = {}
        # TODO we only have to update the collision constraints, if the robot changed
        self.add_collision_avoidance_soft_constraints()

        try:
            self.parse_constraints(move_cmd)
        except AttributeError:
            self.raise_to_blackboard(InsolvableException(u'couldn\'t transform goal'))
            traceback.print_exc()
            return Status.SUCCESS
        except InsolvableException as e:
            self.raise_to_blackboard(e)
            traceback.print_exc()
            return Status.SUCCESS
        except Exception as e:
            self.raise_to_blackboard(e)
            traceback.print_exc()
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
            try:
                C = self.allowed_constraint_types[constraint.type]
            except KeyError:
                # TODO return next best constraint type
                available_constraints = '\n'.join([x for x in self.allowed_constraint_types.keys()]) + '\n'
                raise InsolvableException(u'unknown constraint {}. available constraint types:\n{}'.format(constraint.type ,available_constraints))

            try:
                if hasattr(constraint, u'parameter_value_pair'):
                    params = json.loads(constraint.parameter_value_pair)
                else:
                    params = convert_ros_message_to_dictionary(constraint)
                    del params[u'type']
                c = C(self.god_map, **params)
                soft_constraints = c.get_constraints()
                self.soft_constraints.update(soft_constraints)
            except:
                traceback.print_exc()
                doc_string = C.make_constraints.__doc__
                if doc_string is None:
                    doc_string = 'there is no documentation for this function'
                raise ImplementationException(doc_string)

    def add_js_controller_soft_constraints(self):
        for joint_name in self.get_robot().controlled_joints:
            c = JointPosition(self.get_god_map(), joint_name, self.get_robot().joint_state[joint_name].position, 0, 0)
            self.soft_constraints.update(c.make_constraints())

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
        for joint_name in self.get_robot().controlled_joints:
            for i in range(self.get_god_map().safe_get_data(identifier.number_of_repeller)):
                constraint = ExternalCollisionAvoidance(self.god_map, joint_name,
                                                    max_weight_distance=self.get_god_map().safe_get_data(
                                                    identifier.distance_thresholds +
                                                    [joint_name, u'max_weight_distance']),
                                                    low_weight_distance=self.get_god_map().safe_get_data(
                                                    identifier.distance_thresholds +
                                                    [joint_name, u'low_weight_distance']),
                                                    zero_weight_distance=self.get_god_map().safe_get_data(
                                                    identifier.distance_thresholds +
                                                    [joint_name, u'zero_weight_distance']),
                                                    idx=i)
                soft_constraints.update(constraint.get_constraints())
        for link_a, link_b in self.get_robot().get_self_collision_matrix():
            if not self.get_robot().link_order(link_a, link_b):
                tmp = link_a
                link_a = link_b
                link_b = tmp
            max_weight_distance = min(self.get_god_map().safe_get_data(identifier.distance_thresholds +
                                                                       [link_a, u'max_weight_distance']),
                                      self.get_god_map().safe_get_data(identifier.distance_thresholds +
                                                                       [link_b, u'max_weight_distance']))
            low_weight_distance = min(self.get_god_map().safe_get_data(identifier.distance_thresholds +
                                                                       [link_a, u'low_weight_distance']),
                                      self.get_god_map().safe_get_data(identifier.distance_thresholds +
                                                                       [link_b, u'low_weight_distance']))
            zero_weight_distance = min(self.get_god_map().safe_get_data(identifier.distance_thresholds +
                                                                       [link_a, u'zero_weight_distance']),
                                      self.get_god_map().safe_get_data(identifier.distance_thresholds +
                                                                       [link_b, u'zero_weight_distance']))
            constraint = SelfCollisionAvoidance(self.god_map, link_a, link_b,
                                                max_weight_distance=max_weight_distance,
                                                low_weight_distance=low_weight_distance,
                                                zero_weight_distance=zero_weight_distance)
            soft_constraints.update(constraint.get_constraints())

        self.soft_constraints.update(soft_constraints)
