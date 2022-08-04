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
from giskardpy.configs.data_types import CollisionCheckerLib
from giskardpy.exceptions import UnknownConstraintException, InvalidGoalException, \
    ConstraintInitalizationException, GiskardException
from giskardpy.goals.collision_avoidance import SelfCollisionAvoidance, ExternalCollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.tree.behaviors.get_goal import GetGoal
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import convert_dictionary_to_ros_message, get_all_classes_in_package, raise_to_blackboard, \
    catch_and_raise_to_blackboard


class RosMsgToGoal(GetGoal):
    # FIXME no error msg when constraint has missing parameter
    @profile
    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)
        self.allowed_constraint_types = get_all_classes_in_package(giskardpy.goals, Goal)

    @profile
    def initialise(self):
        self.clear_blackboard_exception()

    @profile
    @catch_and_raise_to_blackboard
    def update(self):
        # TODO make this interruptable
        loginfo('Parsing goal message.')
        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            return Status.FAILURE
        self.get_god_map().set_data(identifier.goals, {})
        try:
            self.parse_constraints(move_cmd)
        except AttributeError:
            raise_to_blackboard(InvalidGoalException('Couldn\'t transform goal'))
            traceback.print_exc()
            return Status.SUCCESS
        except Exception as e:
            raise_to_blackboard(e)
            traceback.print_exc()
            return Status.SUCCESS
        if self.god_map.get_data(identifier.collision_checker) != CollisionCheckerLib.none:
            self.parse_collision_entries(move_cmd.collisions)
        loginfo('Done parsing goal message.')
        return Status.SUCCESS

    @profile
    def parse_constraints(self, cmd: MoveCmd):
        for constraint in itertools.chain(cmd.constraints):
            try:
                loginfo(f'Adding constraint of type: \'{constraint.type}\'')
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
                        f'unknown constraint {constraint.type}. did you mean one of these?:\n{matches}')
                else:
                    available_constraints = '\n'.join([x for x in self.allowed_constraint_types.keys()]) + '\n'
                    raise UnknownConstraintException(
                        f'unknown constraint {constraint.type}. available constraint types:\n{available_constraints}')

            try:
                parsed_json = json.loads(constraint.parameter_value_pair)
                params = self.replace_jsons_with_ros_messages(parsed_json)
                c: Goal = C(god_map=self.god_map, **params)
                c.save_self_on_god_map()
            except Exception as e:
                traceback.print_exc()
                doc_string = C.__init__.__doc__
                error_msg = f'Initialization of \'{C.__name__}\' constraint failed: \n {e} \n'
                if doc_string is not None:
                    error_msg = error_msg + doc_string
                if not isinstance(e, GiskardException):
                    raise ConstraintInitalizationException(error_msg)
                raise e

    def replace_jsons_with_ros_messages(self, d):
        # TODO find message type
        if isinstance(d, list):
            result = list()
            for i, element in enumerate(d):
                result.append(self.replace_jsons_with_ros_messages(element))
            return result
        elif isinstance(d, dict):
            if 'message_type' in d:
                return convert_dictionary_to_ros_message(d)
            else:
                result = {}
                for key, value in d.items():
                    result[key] = self.replace_jsons_with_ros_messages(value)
                return result
        return d

    @profile
    def parse_collision_entries(self, collision_entries: List[CollisionEntry]):
        """
        Adds a constraint for each link that pushed it away from its closest point.
        """
        # FIXME this only catches the most obvious cases
        collision_matrix = self.collision_entries_to_collision_matrix(collision_entries)
        self.god_map.set_data(identifier.collision_matrix, collision_matrix)
        self.time_collector.collision_avoidance.append(0)
        if not collision_entries or not self.collision_scene.is_allow_all_collision(collision_entries[-1]):
            self.add_external_collision_avoidance_constraints(soft_threshold_override=collision_matrix)
        if not collision_entries or (not self.collision_scene.is_allow_all_collision(collision_entries[-1]) and
                                     not self.collision_scene.is_allow_all_self_collision(collision_entries[-1])):
            self.add_self_collision_avoidance_constraints()

    def collision_entries_to_collision_matrix(self, collision_entries: List[CollisionEntry]):
        self.collision_scene.sync()
        max_distances = self.make_max_distances()
        collision_matrix = self.collision_scene.collision_goals_to_collision_matrix(deepcopy(collision_entries),
                                                                                    max_distances)
        return collision_matrix

    def _cal_max_param(self, parameter_name):
        external_distances = self.get_god_map().get_data(identifier.external_collision_avoidance)
        self_distances = self.get_god_map().get_data(identifier.self_collision_avoidance)
        default_distance = max(getattr(external_distances.default_factory(), parameter_name),
                               getattr(self_distances.default_factory(), parameter_name))
        for value in external_distances.values():
            default_distance = max(default_distance, getattr(value, parameter_name))
        for value in self_distances.values():
            default_distance = max(default_distance, getattr(value, parameter_name))
        return default_distance

    def make_max_distances(self):
        external_distances = self.get_god_map().get_data(identifier.external_collision_avoidance)
        self_distances = self.get_god_map().get_data(identifier.self_collision_avoidance)
        # FIXME check all dict entries
        default_distance = self._cal_max_param('soft_threshold')

        max_distances = defaultdict(lambda: default_distance)
        # override max distances based on external distances dict
        for link_name in self.robot.link_names_with_collisions:
            try:
                controlled_parent_joint = self.world.get_controlled_parent_joint_of_link(link_name)
                distance = external_distances[controlled_parent_joint].soft_threshold
                for child_link_name in self.world.get_directly_controlled_child_links_with_collisions(
                        controlled_parent_joint):
                    max_distances[child_link_name] = distance
            except KeyError:
                pass

        for link_name in self_distances:
            distance = self_distances[link_name].soft_threshold
            if link_name in max_distances:
                max_distances[link_name] = max(distance, max_distances[link_name])
            else:
                max_distances[link_name] = distance

        return max_distances

    @profile
    def add_external_collision_avoidance_constraints(self, soft_threshold_override=None):
        controlled_joints = self.god_map.get_data(identifier.controlled_joints)
        config = self.get_god_map().get_data(identifier.external_collision_avoidance)
        for joint_name in controlled_joints:
            child_links = self.world.get_directly_controlled_child_links_with_collisions(joint_name)
            if child_links:
                number_of_repeller = config[joint_name].number_of_repeller
                for i in range(number_of_repeller):
                    child_link = self.world.joints[joint_name].child_link_name
                    hard_threshold = config[joint_name].hard_threshold
                    if soft_threshold_override is not None:
                        soft_threshold = soft_threshold_override
                    else:
                        soft_threshold = config[joint_name].soft_threshold
                    constraint = ExternalCollisionAvoidance(god_map=self.god_map,
                                                            link_name=child_link,
                                                            hard_threshold=hard_threshold,
                                                            soft_thresholds=soft_threshold,
                                                            idx=i,
                                                            num_repeller=number_of_repeller)
                    constraint.save_self_on_god_map()
                    self.time_collector.collision_avoidance[-1] += 1
        loginfo(f'Adding {self.time_collector.collision_avoidance[-1]} self collision avoidance constraints.')

    @profile
    def add_self_collision_avoidance_constraints(self):
        counter = defaultdict(int)
        config = self.get_god_map().get_data(identifier.self_collision_avoidance)
        robot_group_name = self.god_map.unsafe_get_data(identifier.robot_group_name)
        for link_a_o, link_b_o in self.world.groups[robot_group_name].possible_collision_combinations():
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
            num_of_constraints = min(1, counter[link_a, link_b])
            for i in range(num_of_constraints):
                key = f'{link_a}, {link_b}'
                key_r = f'{link_b}, {link_a}'
                # FIXME there is probably a bug or unintuitive behavior, when a pair is affected by multiple entries
                if key in config:
                    hard_threshold = config[key].hard_threshold
                    soft_threshold = config[key].soft_threshold
                    number_of_repeller = config[key].number_of_repeller
                elif key_r in config:
                    hard_threshold = config[key_r].hard_threshold
                    soft_threshold = config[key_r].soft_threshold
                    number_of_repeller = config[key_r].number_of_repeller
                else:
                    # TODO minimum is not the best if i reduce to the links next to the controlled chains
                    #   should probably add symbols that retrieve the values for the current pair
                    hard_threshold = min(config[link_a].hard_threshold,
                                         config[link_b].hard_threshold)
                    soft_threshold = min(config[link_a].soft_threshold,
                                         config[link_b].soft_threshold)
                    number_of_repeller = min(config[link_a].number_of_repeller,
                                             config[link_b].number_of_repeller)
                constraint = SelfCollisionAvoidance(god_map=self.god_map,
                                                    link_a=link_a,
                                                    link_b=link_b,
                                                    hard_threshold=hard_threshold,
                                                    soft_threshold=soft_threshold,
                                                    idx=i,
                                                    num_repeller=number_of_repeller)
                constraint.save_self_on_god_map()
                self.time_collector.collision_avoidance[-1] += 1
        loginfo(f'Adding {self.time_collector.collision_avoidance[-1]} self collision avoidance constraints.')
