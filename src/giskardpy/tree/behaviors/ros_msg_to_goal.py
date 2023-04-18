import difflib
import itertools
import json
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple

from py_trees import Status

import giskardpy.identifier as identifier
from giskard_msgs.msg import MoveCmd, CollisionEntry
from giskardpy.configs.data_types import CollisionCheckerLib
from giskardpy.exceptions import UnknownConstraintException, InvalidGoalException, \
    ConstraintInitalizationException, GiskardException
from giskardpy.goals.collision_avoidance import SelfCollisionAvoidance, ExternalCollisionAvoidance
from giskardpy.goals.goal import Goal
from giskardpy.my_types import PrefixName
from giskardpy.tree.behaviors.get_goal import GetGoal
from giskardpy.utils.logging import loginfo
from giskardpy.utils.utils import convert_dictionary_to_ros_message, get_all_classes_in_package, raise_to_blackboard
from giskardpy.utils.decorators import catch_and_raise_to_blackboard, record_time


class RosMsgToGoal(GetGoal):
    @record_time
    @profile
    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)
        goal_package_paths = self.god_map.get_data(identifier.giskard).goal_package_paths
        self.allowed_constraint_types = {}
        for path in goal_package_paths:
            self.allowed_constraint_types.update(get_all_classes_in_package(path, Goal))
        self.robot_names = self.collision_scene.robot_names

    @record_time
    @profile
    def initialise(self):
        self.clear_blackboard_exception()

    @catch_and_raise_to_blackboard
    @record_time
    @profile
    def update(self):
        loginfo('Parsing goal message.')
        move_cmd = self.god_map.get_data(identifier.next_move_goal)  # type: MoveCmd
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
            # traceback.print_exc()
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
                c: Goal = C(**params)
                c._save_self_on_god_map()
            except Exception as e:
                traceback.print_exc()
                doc_string = C.__init__.__doc__
                error_msg = f'Initialization of \'{C.__name__}\' constraint failed: \n {e} \n'
                # if doc_string is not None:
                #     error_msg = error_msg + doc_string
                if not isinstance(e, GiskardException):
                    raise ConstraintInitalizationException(error_msg)
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
        collision_matrix = self.collision_entries_to_collision_matrix(collision_entries)
        self.god_map.set_data(identifier.collision_matrix, collision_matrix)
        if not collision_entries or not self.collision_scene.is_allow_all_collision(collision_entries[-1]):
            self.add_external_collision_avoidance_constraints(soft_threshold_override=collision_matrix)
        if not collision_entries or (not self.collision_scene.is_allow_all_collision(collision_entries[-1]) and
                                     not self.collision_scene.is_allow_all_self_collision(collision_entries[-1])):
            self.add_self_collision_avoidance_constraints()

    def collision_entries_to_collision_matrix(self, collision_entries: List[CollisionEntry]):
        self.collision_scene.sync()
        max_distances = self.make_max_distances()
        # ignored_collisions = self.collision_scene.ignored_self_collion_pairs
        collision_matrix = self.collision_scene.collision_goals_to_collision_matrix(deepcopy(collision_entries),
                                                                                    max_distances)
        return collision_matrix

    def make_max_distances(self) -> Dict[Tuple[PrefixName, PrefixName], float]:
        default_distance = {}
        # fixme this default is buggy, but it doesn't get triggered
        for robot_name in self.robot_names:
            collision_avoidance_config = self.collision_avoidance_configs[robot_name]
            external_distances = collision_avoidance_config.external_collision_avoidance
            self_distances = collision_avoidance_config.self_collision_avoidance
            default_distance[robot_name] = collision_avoidance_config.cal_max_param('soft_threshold')

        max_distances = defaultdict(lambda: default_distance)
        # override max distances based on external distances dict
        for robot in self.collision_scene.robots:
            for link_name in robot.link_names_with_collisions:
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
        configs = self.collision_avoidance_configs
        fixed_joints = self.collision_scene.fixed_joints
        joints = [j for j in self.world.controlled_joints if j not in fixed_joints]
        num_constrains = 0
        for joint_name in joints:
            try:
                robot_name = self.world.get_group_of_joint(joint_name).name
            except KeyError:
                child_link = self.world.joints[joint_name].child_link_name
                robot_name = self.world._get_group_name_containing_link(child_link)
            child_links = self.world.get_directly_controlled_child_links_with_collisions(joint_name, fixed_joints)
            if child_links:
                number_of_repeller = configs[robot_name].external_collision_avoidance[joint_name].number_of_repeller
                for i in range(number_of_repeller):
                    child_link = self.world.joints[joint_name].child_link_name
                    hard_threshold = configs[robot_name].external_collision_avoidance[joint_name].hard_threshold
                    if soft_threshold_override is not None:
                        soft_threshold = soft_threshold_override
                    else:
                        soft_threshold = configs[robot_name].external_collision_avoidance[joint_name].soft_threshold
                    constraint = ExternalCollisionAvoidance(robot_name=robot_name,
                                                            link_name=child_link,
                                                            hard_threshold=hard_threshold,
                                                            soft_thresholds=soft_threshold,
                                                            idx=i,
                                                            num_repeller=number_of_repeller)
                    constraint._save_self_on_god_map()
                    num_constrains += 1
        loginfo(f'Adding {num_constrains} external collision avoidance constraints.')

    @profile
    def add_self_collision_avoidance_constraints(self):
        counter = defaultdict(int)
        fixed_joints = self.collision_scene.fixed_joints
        configs = self.collision_avoidance_configs
        num_constr = 0
        for robot_name in self.robot_names:
            for link_a_o, link_b_o in self.world.groups[robot_name].possible_collision_combinations():
                link_a_o, link_b_o = self.world.sort_links(link_a_o, link_b_o)
                try:
                    if (link_a_o, link_b_o) in self.collision_scene.black_list:
                        continue
                    link_a, link_b = self.world.compute_chain_reduced_to_controlled_joints(link_a_o, link_b_o, fixed_joints)
                    link_a, link_b = self.world.sort_links(link_a, link_b)
                    counter[link_a, link_b] += 1
                except KeyError as e:
                    # no controlled joint between both links
                    pass

        for link_a, link_b in counter:
            group_names = self.world.get_group_names_containing_link(link_a)
            if len(group_names) != 1:
                group_name = self.world.get_parent_group_name(group_names.pop())
            else:
                group_name = group_names.pop()
            num_of_constraints = min(1, counter[link_a, link_b])
            for i in range(num_of_constraints):
                key = f'{link_a}, {link_b}'
                key_r = f'{link_b}, {link_a}'
                config = configs[group_name].self_collision_avoidance
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
                groups_a = self.world._get_group_name_containing_link(link_a)
                groups_b = self.world._get_group_name_containing_link(link_b)
                if groups_b == groups_a:
                    robot_name = groups_a
                else:
                    raise Exception(f'Could not find group containing the link {link_a} and {link_b}.')
                constraint = SelfCollisionAvoidance(link_a=link_a,
                                                    link_b=link_b,
                                                    robot_name=robot_name,
                                                    hard_threshold=hard_threshold,
                                                    soft_threshold=soft_threshold,
                                                    idx=i,
                                                    num_repeller=number_of_repeller)
                constraint._save_self_on_god_map()
                num_constr += 1
        loginfo(f'Adding {num_constr} self collision avoidance constraints.')
