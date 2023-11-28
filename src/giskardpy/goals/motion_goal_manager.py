import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple

from giskardpy.exceptions import UnknownConstraintException, GiskardException, ConstraintInitalizationException, \
    DuplicateNameException
from giskardpy.goals.collision_avoidance import ExternalCollisionAvoidance, SelfCollisionAvoidance
from giskardpy.goals.goal import Goal
import giskard_msgs.msg as giskard_msgs
from giskardpy.god_map import god_map
from giskardpy.my_types import PrefixName
from giskardpy.utils import logging
from giskardpy.utils.utils import get_all_classes_in_package, json_to_kwargs, convert_dictionary_to_ros_message, \
    json_str_to_kwargs, ImmutableDict


class MotionGoalManager:
    motion_goals: Dict[str, Goal] = None

    def __init__(self):
        self.motion_goals = {}
        goal_package_paths = god_map.giskard.goal_package_paths
        self.allowed_motion_goal_types = {}
        for path in goal_package_paths:
            self.allowed_motion_goal_types.update(get_all_classes_in_package(path, Goal))
        self.robot_names = god_map.collision_scene.robot_names

    @profile
    def parse_motion_goals(self, motion_goals: List[giskard_msgs.MotionGoal]):
        for motion_goal in motion_goals:
            try:
                logging.loginfo(f'Adding motion goal of type: \'{motion_goal.type}\'')
                C = self.allowed_motion_goal_types[motion_goal.type]
            except KeyError:
                raise UnknownConstraintException(f'unknown constraint {motion_goal.type}.')
            try:
                params = json_str_to_kwargs(motion_goal.parameter_value_pair)
                if motion_goal.name == '':
                    motion_goal.name = None
                to_start = [god_map.monitor_manager.get_monitor(monitor_name) for monitor_name in motion_goal.to_start]
                to_hold = [god_map.monitor_manager.get_monitor(monitor_name) for monitor_name in motion_goal.to_hold]
                to_end = [god_map.monitor_manager.get_monitor(monitor_name) for monitor_name in motion_goal.to_end]
                c: Goal = C(name=motion_goal.name, to_start=to_start, to_hold=to_hold, to_end=to_end, **params)
                self.add_motion_goal(c)
            except Exception as e:
                traceback.print_exc()
                error_msg = f'Initialization of \'{C.__name__}\' constraint failed: \n {e} \n'
                if not isinstance(e, GiskardException):
                    raise ConstraintInitalizationException(error_msg)
                raise e

    def add_motion_goal(self, goal: Goal):
        name = goal.name
        if name in self.motion_goals:
            raise DuplicateNameException(f'Motion goal with name {name} already exists.')
        self.motion_goals[name] = goal

    @profile
    def parse_collision_entries(self, collision_entries: List[giskard_msgs.CollisionEntry]):
        """
        Adds a constraint for each link that pushed it away from its closest point.
        """
        god_map.collision_scene.create_collision_matrix(deepcopy(collision_entries))
        if not collision_entries or not god_map.collision_scene.is_allow_all_collision(collision_entries[-1]):
            self.add_external_collision_avoidance_constraints(
                soft_threshold_override=god_map.collision_scene.collision_matrix)
        if not collision_entries or (not god_map.collision_scene.is_allow_all_collision(collision_entries[-1]) and
                                     not god_map.collision_scene.is_allow_all_self_collision(collision_entries[-1])):
            self.add_self_collision_avoidance_constraints()

    @profile
    def add_external_collision_avoidance_constraints(self, soft_threshold_override=None):
        configs = god_map.collision_scene.collision_avoidance_configs
        fixed_joints = god_map.collision_scene.fixed_joints
        joints = [j for j in god_map.world.controlled_joints if j not in fixed_joints]
        num_constrains = 0
        for joint_name in joints:
            try:
                robot_name = god_map.world.get_group_of_joint(joint_name).name
            except KeyError:
                child_link = god_map.world.joints[joint_name].child_link_name
                robot_name = god_map.world._get_group_name_containing_link(child_link)
            child_links = god_map.world.get_directly_controlled_child_links_with_collisions(joint_name, fixed_joints)
            if child_links:
                number_of_repeller = configs[robot_name].external_collision_avoidance[joint_name].number_of_repeller
                for i in range(number_of_repeller):
                    child_link = god_map.world.joints[joint_name].child_link_name
                    hard_threshold = configs[robot_name].external_collision_avoidance[joint_name].hard_threshold
                    if soft_threshold_override is not None:
                        soft_threshold = soft_threshold_override
                    else:
                        soft_threshold = configs[robot_name].external_collision_avoidance[joint_name].soft_threshold
                    motion_goal = ExternalCollisionAvoidance(robot_name=robot_name,
                                                             link_name=child_link,
                                                             hard_threshold=hard_threshold,
                                                             soft_thresholds=soft_threshold,
                                                             idx=i,
                                                             num_repeller=number_of_repeller)
                    god_map.motion_goal_manager.add_motion_goal(motion_goal)
                    num_constrains += 1
        logging.loginfo(f'Adding {num_constrains} external collision avoidance constraints.')

    @profile
    def add_self_collision_avoidance_constraints(self):
        counter = defaultdict(int)
        fixed_joints = god_map.collision_scene.fixed_joints
        configs = god_map.collision_scene.collision_avoidance_configs
        num_constr = 0
        for robot_name in self.robot_names:
            for link_a_o, link_b_o in god_map.world.groups[robot_name].possible_collision_combinations():
                link_a_o, link_b_o = god_map.world.sort_links(link_a_o, link_b_o)
                try:
                    if (link_a_o, link_b_o) in god_map.collision_scene.self_collision_matrix:
                        continue
                    link_a, link_b = god_map.world.compute_chain_reduced_to_controlled_joints(link_a_o, link_b_o,
                                                                                              fixed_joints)
                    link_a, link_b = god_map.world.sort_links(link_a, link_b)
                    counter[link_a, link_b] += 1
                except KeyError as e:
                    # no controlled joint between both links
                    pass

        for link_a, link_b in counter:
            group_names = god_map.world.get_group_names_containing_link(link_a)
            if len(group_names) != 1:
                group_name = god_map.world.get_parent_group_name(group_names.pop())
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
                groups_a = god_map.world._get_group_name_containing_link(link_a)
                groups_b = god_map.world._get_group_name_containing_link(link_b)
                if groups_b == groups_a:
                    robot_name = groups_a
                else:
                    raise Exception(f'Could not find group containing the link {link_a} and {link_b}.')
                goal = SelfCollisionAvoidance(link_a=link_a,
                                              link_b=link_b,
                                              robot_name=robot_name,
                                              hard_threshold=hard_threshold,
                                              soft_threshold=soft_threshold,
                                              idx=i,
                                              num_repeller=number_of_repeller)
                self.add_motion_goal(goal)
                num_constr += 1
        logging.loginfo(f'Adding {num_constr} self collision avoidance constraints.')

    @profile
    def get_constraints_from_goals(self):
        eq_constraints = ImmutableDict()
        neq_constraints = ImmutableDict()
        derivative_constraints = ImmutableDict()
        manip_constraints = ImmutableDict()
        goals: Dict[str, Goal] = god_map.motion_goal_manager.motion_goals
        for goal_name, goal in list(goals.items()):
            try:
                new_eq_constraints, new_neq_constraints, new_derivative_constraints, _debug_expressions, new_manip_constraints = goal.get_constraints()
            except Exception as e:
                raise ConstraintInitalizationException(str(e))
            eq_constraints.update(new_eq_constraints)
            neq_constraints.update(new_neq_constraints)
            derivative_constraints.update(new_derivative_constraints)
            manip_constraints.update(new_manip_constraints)
            # logging.loginfo(f'{goal_name} added {len(_constraints)+len(_vel_constraints)} constraints.')
        god_map.eq_constraints = eq_constraints
        god_map.neq_constraints = neq_constraints
        god_map.derivative_constraints = derivative_constraints
        god_map.manip_constraints = manip_constraints
        return eq_constraints, neq_constraints, derivative_constraints, manip_constraints

    def replace_jsons_with_ros_messages(self, d):
        if isinstance(d, list):
            for i, element in enumerate(d):
                d[i] = self.replace_jsons_with_ros_messages(element)

        if isinstance(d, dict):

            if 'message_type' in d:

                d = convert_dictionary_to_ros_message(d)

            else:
                for key, value in d.copy().items():
                    d[key] = self.replace_jsons_with_ros_messages(value)

        return d
