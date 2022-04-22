import difflib
import itertools
import json
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List

from py_trees import Status

import giskardpy.goals
import giskardpy.identifier as identifier
from giskard_msgs.msg import MoveCmd, CollisionEntry
from giskardpy import casadi_wrapper as w
from giskardpy.data_types import PrefixName
from giskardpy.exceptions import UnknownConstraintException, InvalidGoalException, \
    ConstraintInitalizationException, GiskardException
from giskardpy.goals.collision_avoidance import SelfCollisionAvoidance, ExternalCollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.tree.behaviors.get_goal import GetGoal
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import convert_dictionary_to_ros_message, get_all_classes_in_package


class GoalToConstraints(GetGoal):
    # FIXME no error msg when constraint has missing parameter
    @profile
    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)
        self.used_joints = set()

        self.controlled_joints = set()
        self.controllable_links = set()
        self.last_urdf = None
        self.allowed_constraint_types = get_all_classes_in_package(giskardpy.goals, Goal)
        self.robot_names = self.collision_scene.robot_names

        self.rc_prismatic_velocity = self.get_god_map().get_data(identifier.rc_prismatic_velocity)
        self.rc_continuous_velocity = self.get_god_map().get_data(identifier.rc_continuous_velocity)
        self.rc_revolute_velocity = self.get_god_map().get_data(identifier.rc_revolute_velocity)
        self.rc_other_velocity = self.get_god_map().get_data(identifier.rc_other_velocity)

    @profile
    def initialise(self):
        self.clear_blackboard_exception()

    @profile
    def update(self):
        # TODO make this interruptable
        try:
            loginfo('Parsing goal message.')
            move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
            if not move_cmd:
                return Status.FAILURE

            self.get_god_map().set_data(identifier.goals, {})

            self.soft_constraints = {}
            self.vel_constraints = {}
            self.debug_expr = {}

            try:
                self.parse_constraints(move_cmd)
            except AttributeError:
                self.raise_to_blackboard(InvalidGoalException('couldn\'t transform goal'))
                traceback.print_exc()
                return Status.SUCCESS
            except Exception as e:
                self.raise_to_blackboard(e)
                traceback.print_exc()
                return Status.SUCCESS

            if not (self.get_god_map().get_data(identifier.check_reachability)) and \
                    self.god_map.get_data(identifier.collision_checker) is not None:
                self.parse_collision_entries(move_cmd.collisions)

            self.get_god_map().set_data(identifier.constraints, self.soft_constraints)
            self.get_god_map().set_data(identifier.vel_constraints, self.vel_constraints)
            self.get_god_map().set_data(identifier.debug_expressions, self.debug_expr)

            if self.get_god_map().get_data(identifier.check_reachability):
                self.raise_to_blackboard(NotImplementedError('reachability check is not implemented'))
                return Status.SUCCESS

            l = self.active_free_symbols()
            free_variables = list(sorted([v for v in self.world.joint_constraints if v.name in l],
                                         key=lambda x: x.name))
            self.get_god_map().set_data(identifier.free_variables, free_variables)
            loginfo('Done parsing goal message.')
            return Status.SUCCESS
        except Exception as e:
            traceback.print_exc()
            self.raise_to_blackboard(e)
            return Status.FAILURE

    def active_free_symbols(self):
        symbols = set()
        for c in self.soft_constraints.values():
            symbols.update(str(s) for s in w.free_symbols(c.expression))
        return symbols

    @profile
    def parse_constraints(self, cmd):
        """
        :type cmd: MoveCmd
        :rtype: dict
        """
        for constraint in itertools.chain(cmd.constraints):
            try:
                loginfo('Adding constraint of type: \'{}\''.format(constraint.type))
                C = self.allowed_constraint_types[constraint.type]
            except KeyError:
                matches = ''
                for s in self.allowed_constraint_types.keys():
                    sm = difflib.SequenceMatcher(None, str(constraint.type).lower(), s.lower())
                    ratio = sm.ratio()
                    if ratio >= 0.5:
                        matches = matches + s + '\n'
                if matches != '':
                    raise UnknownConstraintException(
                        'unknown constraint {}. did you mean one of these?:\n{}'.format(constraint.type, matches))
                else:
                    available_constraints = '\n'.join([x for x in self.allowed_constraint_types.keys()]) + '\n'
                    raise UnknownConstraintException(
                        'unknown constraint {}. available constraint types:\n{}'.format(constraint.type,
                                                                                        available_constraints))

            try:
                parsed_json = json.loads(constraint.parameter_value_pair)
                params = self.replace_jsons_with_ros_messages(parsed_json)
                c = C(god_map=self.god_map, **params)
            except Exception as e:
                traceback.print_exc()
                doc_string = C.__init__.__doc__
                error_msg = 'Initialization of "{}" constraint failed: \n {} \n'.format(C.__name__, e)
                if doc_string is not None:
                    error_msg = error_msg + doc_string
                if not isinstance(e, GiskardException):
                    raise ConstraintInitalizationException(error_msg)
                raise e
            try:
                soft_constraints, vel_constraints, debug_expressions = c.get_constraints()
                self.soft_constraints.update(soft_constraints)
                self.vel_constraints.update(vel_constraints)
                self.debug_expr.update(debug_expressions)
            except Exception as e:
                traceback.print_exc()
                if not isinstance(e, GiskardException):
                    raise ConstraintInitalizationException(e)
                raise e

    def replace_jsons_with_ros_messages(self, d):
        for key, value in d.items():
            if isinstance(value, dict) and 'message_type' in value:
                d[key] = convert_dictionary_to_ros_message(value)
        return d

    @profile
    def parse_collision_entries(self, collision_entries: List[CollisionEntry]):
        """
        Adds a constraint for each link that pushed it away from its closest point.
        """
        # FIXME this only catches the most obvious cases
        collision_matrix = self.collision_entries_to_collision_matrix(collision_entries)
        self.god_map.set_data(identifier.collision_matrix, collision_matrix)
        soft_threshold = None
        for collision_cmd in collision_entries:
            if self.collision_scene.is_avoid_all_collision(collision_cmd):
                soft_threshold = collision_cmd.distance
        self.time_collector.collision_avoidance.append(0)
        if not collision_entries or not self.collision_scene.is_allow_all_collision(collision_entries[-1]):
            self.add_external_collision_avoidance_constraints(soft_threshold_override=collision_matrix)
        if not collision_entries or (not self.collision_scene.is_allow_all_collision(collision_entries[-1]) and
                                     not self.collision_scene.is_allow_all_self_collision(collision_entries[-1])):
            self.add_self_collision_avoidance_constraints()

    def collision_entries_to_collision_matrix(self, collision_entries: List[CollisionEntry]):
        # t = time()
        self.collision_scene.sync()
        max_distances = self.make_max_distances()
        try:
            added_checks = self.get_god_map().get_data(identifier.added_collision_checks)
            self.god_map.set_data(identifier.added_collision_checks, {})
        except KeyError:
            # no collision checks added
            added_checks = {}
        collision_matrix = self.collision_scene.collision_goals_to_collision_matrix(deepcopy(collision_entries),
                                                                                    max_distances,
                                                                                    added_checks)
        # t2 = time() - t
        # self.get_blackboard().runtime += t2
        return collision_matrix

    def _cal_max_param(self, parameter_name):
        external_distances = self.get_god_map().get_data(identifier.external_collision_avoidance)
        self_distances = self.get_god_map().get_data(identifier.self_collision_avoidance)
        try:
            default_distance = max(external_distances.default_factory(parameter_name.prefix)[parameter_name],
                                   self_distances.default_factory(parameter_name.prefix)[parameter_name])
        except Exception:
            pass
        for value in external_distances.values():
            default_distance = max(default_distance, value[parameter_name])
        for value in self_distances.values():
            default_distance = max(default_distance, value[parameter_name])
        return default_distance

    def make_max_distances(self):
        external_distances = self.get_god_map().get_data(identifier.external_collision_avoidance)
        self_distances = self.get_god_map().get_data(identifier.self_collision_avoidance)
        # FIXME check all dict entries
        default_distance = {r_n: self._cal_max_param(PrefixName('soft_threshold', r_n)) for r_n in
                            self.robot_names}

        max_distances = defaultdict(lambda: default_distance)
        # override max distances based on external distances dict
        for robot in self.collision_scene.robots:
            for link_name in robot.link_names_with_collisions:
                controlled_parent_joint = robot.get_controlled_parent_joint_of_link(link_name)
                distance = external_distances[controlled_parent_joint][PrefixName('soft_threshold', robot.name)]
                for child_link_name in robot.get_directly_controlled_child_links_with_collisions(
                        controlled_parent_joint):
                    max_distances[child_link_name] = distance

        for link_name in self_distances:
            distance = self_distances[link_name][PrefixName('soft_threshold', link_name.prefix)]
            if link_name in max_distances:
                max_distances[link_name] = max(distance, max_distances[link_name])
            else:
                max_distances[link_name] = distance

        return max_distances

    @profile
    def add_external_collision_avoidance_constraints(self, soft_threshold_override=None):
        soft_constraints = {}
        vel_constraints = {}
        debug_expr = {}
        controlled_joints = self.god_map.get_data(identifier.controlled_joints)
        config = self.get_god_map().get_data(identifier.external_collision_avoidance)
        for joint_name in controlled_joints:
            robot = self.world.get_group_of_joint(joint_name)
            child_links = self.world.groups[robot.name].get_directly_controlled_child_links_with_collisions(joint_name)
            if child_links:
                number_of_repeller = config[joint_name][PrefixName('number_of_repeller', robot.name)]
                for i in range(number_of_repeller):
                    child_link = self.world.joints[joint_name].child_link_name
                    hard_threshold = config[joint_name][PrefixName('hard_threshold', robot.name)]
                    if soft_threshold_override is not None:
                        soft_threshold = soft_threshold_override
                    else:
                        soft_threshold = config[joint_name][PrefixName('soft_threshold', robot.name)]
                    constraint = ExternalCollisionAvoidance(god_map=self.god_map,
                                                            robot_name=robot.name,
                                                            link_name=child_link,
                                                            hard_threshold=hard_threshold,
                                                            soft_thresholds=soft_threshold,
                                                            idx=i,
                                                            num_repeller=number_of_repeller)
                    c, c_vel, debug_expressions = constraint.get_constraints()
                    soft_constraints.update(c)
                    vel_constraints.update(c_vel)
                    debug_expr.update(debug_expressions)

        num_external = len(soft_constraints)
        loginfo(f'adding {num_external} external collision avoidance constraints')
        self.time_collector.collision_avoidance[-1] += num_external
        self.soft_constraints.update(soft_constraints)
        self.vel_constraints.update(vel_constraints)
        self.debug_expr.update(debug_expr)

    @profile
    def add_self_collision_avoidance_constraints(self):
        counter = defaultdict(int)
        soft_constraints = {}
        vel_constraints = {}
        debug_expr = {}
        config = self.get_god_map().get_data(identifier.self_collision_avoidance)
        for robot_name in self.robot_names:
            for link_a_o, link_b_o in self.world.groups[robot_name].possible_collision_combinations():
                link_a_o, link_b_o = self.world.sort_links(link_a_o, link_b_o)
                try:
                    link_a, link_b = self.world.compute_chain_reduced_to_controlled_joints(link_a_o, link_b_o)
                    link_a, link_b = self.world.sort_links(link_a, link_b)
                    if (link_a, link_b) in self.collision_scene.black_list:
                        continue
                    counter[link_a, link_b] += 1
                except KeyError as e:
                    # no controlled joint between both links
                    pass

        for link_a, link_b in counter:
            group_names = self.world.get_groups_containing_link(link_a)
            if len(group_names) != 1:
                group_name = self.world.get_parent_group_name(group_names.pop())
            else:
                group_name = group_names.pop()
            num_of_constraints = min(1, counter[link_a, link_b])
            for i in range(num_of_constraints):
                key = '{}, {}'.format(link_a, link_b)
                key_r = '{}, {}'.format(link_b, link_a)
                # FIXME there is probably a bug or unintuitive behavior, when a pair is affected by multiple entries
                if key in config:
                    hard_threshold = config[key][PrefixName('hard_threshold', group_name)]
                    soft_threshold = config[key][PrefixName('soft_threshold', group_name)]
                    number_of_repeller = config[key][PrefixName('number_of_repeller', group_name)]
                elif key_r in config:
                    hard_threshold = config[key_r][PrefixName('hard_threshold', group_name)]
                    soft_threshold = config[key_r][PrefixName('soft_threshold', group_name)]
                    number_of_repeller = config[key_r][PrefixName('number_of_repeller', group_name)]
                else:
                    # TODO minimum is not the best if i reduce to the links next to the controlled chains
                    #   should probably add symbols that retrieve the values for the current pair
                    hard_threshold = min(config[link_a][PrefixName('hard_threshold', group_name)],
                                         config[link_b][PrefixName('hard_threshold', group_name)])
                    soft_threshold = min(config[link_a][PrefixName('soft_threshold', group_name)],
                                         config[link_b][PrefixName('soft_threshold', group_name)])
                    number_of_repeller = min(config[link_a][PrefixName('number_of_repeller', group_name)],
                                             config[link_b][PrefixName('number_of_repeller', group_name)])
                groups_a = self.world.get_group_containing_link(link_a)
                groups_b = self.world.get_group_containing_link(link_b)
                if groups_b == groups_a:
                    robot_name = groups_a
                else:
                    raise Exception(f'Could not find group containing the link {link_a} and {link_b}.')
                constraint = SelfCollisionAvoidance(god_map=self.god_map,
                                                    link_a=link_a,
                                                    link_b=link_b,
                                                    robot_name=robot_name,
                                                    hard_threshold=hard_threshold,
                                                    soft_threshold=soft_threshold,
                                                    idx=i,
                                                    num_repeller=number_of_repeller)
                c, c_vel, debug_expressions = constraint.get_constraints()
                soft_constraints.update(c)
                vel_constraints.update(c_vel)
                debug_expr.update(debug_expressions)
        loginfo('adding {} self collision avoidance constraints'.format(len(soft_constraints)))
        self.time_collector.collision_avoidance[-1] += len(soft_constraints)
        self.soft_constraints.update(soft_constraints)
        self.vel_constraints.update(vel_constraints)
        self.debug_expr.update(debug_expr)
