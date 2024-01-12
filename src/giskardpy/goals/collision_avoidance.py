from collections import defaultdict
from copy import deepcopy
from typing import Dict, Optional, List
import giskardpy.casadi_wrapper as cas
import giskardpy.utils.tfwrapper as tf
from giskard_msgs.msg import CollisionEntry
from giskardpy.goals.goal import Goal
from giskardpy.monitors.monitors import ExpressionMonitor
from giskardpy.monitors.payload_monitors import CollisionMatrixUpdater
from giskardpy.tasks.task import WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE, Task
from giskardpy.god_map import god_map
from giskardpy.data_types import my_string
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils import logging


class ExternalCollisionAvoidance(Goal):

    def __init__(self,
                 link_name: my_string,
                 robot_name: str,
                 max_velocity: float = 0.2,
                 hard_threshold: float = 0.0,
                 name_prefix: Optional[str] = None,
                 soft_thresholds: Optional[Dict[my_string, float]] = None,
                 idx: int = 0,
                 num_repeller: int = 1,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        """
        Don't use me
        """
        self.max_velocity = max_velocity
        self.hard_threshold = hard_threshold
        self.soft_thresholds = soft_thresholds
        self.num_repeller = num_repeller
        self.link_name = link_name
        self.idx = idx
        name = f'{name_prefix}/{self.__class__.__name__}/{self.link_name}/{self.idx}'
        super().__init__(name)
        self.root = god_map.world.root_link_name
        self.robot_name = robot_name
        self.control_horizon = god_map.qp_controller_config.prediction_horizon - (
                god_map.qp_controller_config.max_derivative - 1)
        self.control_horizon = max(1, self.control_horizon)

        a_P_pa = self.get_closest_point_on_a_in_a()
        map_V_n = self.map_V_n_symbol()
        actual_distance = self.get_actual_distance()
        sample_period = god_map.qp_controller_config.sample_period
        number_of_external_collisions = self.get_number_of_external_collisions()

        map_T_a = god_map.world.compose_fk_expression(self.root, self.link_name)

        map_P_pa = map_T_a.dot(a_P_pa)

        # the position distance is not accurate, but the derivative is still correct
        dist = map_V_n.dot(map_P_pa)

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        soft_threshold = 0
        actual_link_b_hash = self.get_link_b_hash()
        parent_joint = god_map.world.links[self.link_name].parent_joint_name
        direct_children = set(god_map.world.get_directly_controlled_child_links_with_collisions(parent_joint))
        b_result_cases = [(k[1].__hash__(), v) for k, v in self.soft_thresholds.items() if k[0] in direct_children]
        soft_threshold = cas.if_eq_cases(a=actual_link_b_hash,
                                         b_result_cases=b_result_cases,
                                         else_result=soft_threshold)

        hard_threshold = cas.min(self.hard_threshold, soft_threshold / 2)
        lower_limit = soft_threshold - actual_distance

        lower_limit_limited = cas.limit(lower_limit,
                                        -qp_limits_for_lba,
                                        qp_limits_for_lba)

        upper_slack = cas.if_greater(actual_distance, hard_threshold,
                                     cas.limit(soft_threshold - hard_threshold,
                                               -qp_limits_for_lba,
                                               qp_limits_for_lba),
                                     lower_limit_limited)
        # undo factor in A
        upper_slack /= (sample_period * self.control_horizon)

        upper_slack = cas.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                     1e4,
                                     cas.max(0, upper_slack))

        # weight = cas.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)

        weight = cas.save_division(WEIGHT_COLLISION_AVOIDANCE,  # divide by number of active repeller per link
                                   cas.min(number_of_external_collisions, self.num_repeller))
        distance_monitor = ExpressionMonitor(f'collision distance {self.name}', plot=False)
        distance_monitor.expression = cas.greater(actual_distance, 50)
        self.add_monitor(distance_monitor)
        task = self.create_and_add_task('stay away')
        task.hold_condition = distance_monitor.get_state_expression()
        task.add_inequality_constraint(reference_velocity=self.max_velocity,
                                       lower_error=lower_limit,
                                       upper_error=float('inf'),
                                       weight=weight,
                                       task_expression=dist,
                                       lower_slack_limit=-float('inf'),
                                       upper_slack_limit=upper_slack)
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)

    def map_V_n_symbol(self):
        expr = f'god_map.closest_point.get_external_collisions(\'{self.link_name}\')[{self.idx}].map_V_n'
        return symbol_manager.get_expr(expr, output_type_hint=cas.Vector3)

    def get_closest_point_on_a_in_a(self):
        expr = f'god_map.closest_point.get_external_collisions(\'{self.link_name}\')[{self.idx}].new_a_P_pa'
        return symbol_manager.get_expr(expr, output_type_hint=cas.Point3)

    def map_P_a_symbol(self):
        expr = f'god_map.closest_point.get_external_collisions(\'{self.link_name}\')[{self.idx}].new_map_P_pa'
        return symbol_manager.get_expr(expr, output_type_hint=cas.Point3)

    def get_actual_distance(self):
        expr = f'god_map.closest_point.get_external_collisions(\'{self.link_name}\')[{self.idx}].contact_distance'
        return symbol_manager.get_symbol(expr)

    def get_link_b_hash(self):
        expr = f'god_map.closest_point.get_external_collisions(\'{self.link_name}\')[{self.idx}].link_b_hash'
        return symbol_manager.get_symbol(expr)

    def get_number_of_external_collisions(self):
        expr = f'god_map.closest_point.get_number_of_external_collisions(\'{self.link_name}\')'
        return symbol_manager.get_symbol(expr)


class SelfCollisionAvoidance(Goal):

    def __init__(self,
                 link_a: my_string,
                 link_b: my_string,
                 robot_name: str,
                 max_velocity: float = 0.2,
                 hard_threshold: float = 0.0,
                 soft_threshold: float = 0.05,
                 idx: float = 0,
                 num_repeller: int = 1,
                 name_prefix: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        self.link_a = link_a
        self.link_b = link_b
        self.max_velocity = max_velocity
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.num_repeller = num_repeller
        self.idx = idx
        if self.link_a.prefix != self.link_b.prefix:
            raise Exception(f'Links {self.link_a} and {self.link_b} have different prefix.')
        name = f'{name_prefix}/{self.__class__.__name__}/{self.link_a}/{self.link_b}/{self.idx}'
        super().__init__(name)
        self.root = god_map.world.root_link_name
        self.robot_name = robot_name
        self.control_horizon = god_map.qp_controller_config.prediction_horizon - (
                god_map.qp_controller_config.max_derivative - 1)
        self.control_horizon = max(1, self.control_horizon)

        hard_threshold = cas.min(self.hard_threshold, self.soft_threshold / 2)
        actual_distance = self.get_actual_distance()
        number_of_self_collisions = self.get_number_of_self_collisions()
        sample_period = god_map.qp_controller_config.sample_period

        # b_T_a2 = god_map.get_world().compose_fk_evaluated_expression(self.link_b, self.link_a)
        b_T_a = god_map.world.compose_fk_expression(self.link_b, self.link_a)
        pb_T_b = self.get_b_T_pb().inverse()
        a_P_pa = self.get_position_on_a_in_a()

        pb_V_n = self.get_contact_normal_in_b()

        pb_P_pa = pb_T_b.dot(b_T_a).dot(a_P_pa)

        dist = pb_V_n.dot(pb_P_pa)

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        lower_limit = self.soft_threshold - actual_distance

        lower_limit_limited = cas.limit(lower_limit,
                                        -qp_limits_for_lba,
                                        qp_limits_for_lba)

        upper_slack = cas.if_greater(actual_distance, hard_threshold,
                                     cas.limit(self.soft_threshold - hard_threshold,
                                               -qp_limits_for_lba,
                                               qp_limits_for_lba),
                                     lower_limit_limited)

        # undo factor in A
        upper_slack /= (sample_period * self.control_horizon)

        upper_slack = cas.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                     1e4,
                                     cas.max(0, upper_slack))

        weight = cas.save_division(WEIGHT_COLLISION_AVOIDANCE,  # divide by number of active repeller per link
                                   cas.min(number_of_self_collisions, self.num_repeller))
        distance_monitor = ExpressionMonitor(f'collision distance {self.name}', plot=False)
        distance_monitor.expression = cas.greater(actual_distance, 50)
        self.add_monitor(distance_monitor)
        task = self.create_and_add_task('stay away')
        task.hold_condition = distance_monitor.get_state_expression()
        task.add_inequality_constraint(reference_velocity=self.max_velocity,
                                       lower_error=lower_limit,
                                       upper_error=float('inf'),
                                       weight=weight,
                                       task_expression=dist,
                                       lower_slack_limit=-float('inf'),
                                       upper_slack_limit=upper_slack)
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)

    def get_contact_normal_in_b(self):
        expr = f'god_map.closest_point.get_self_collisions(\'{self.link_a}\', \'{self.link_b}\')[{self.idx}].new_b_V_n'
        return symbol_manager.get_expr(expr, output_type_hint=cas.Vector3)

    def get_position_on_a_in_a(self):
        expr = f'god_map.closest_point.get_self_collisions(\'{self.link_a}\', \'{self.link_b}\')[{self.idx}].new_a_P_pa'
        return symbol_manager.get_expr(expr, output_type_hint=cas.Point3)

    def get_b_T_pb(self) -> cas.TransMatrix:
        expr = f'god_map.closest_point.get_self_collisions(\'{self.link_a}\', \'{self.link_b}\')[{self.idx}].new_b_P_pb'
        p = symbol_manager.get_expr(expr, output_type_hint=cas.Point3)
        return cas.TransMatrix.from_xyz_rpy(x=p.x, y=p.y, z=p.z)

    def get_actual_distance(self):
        expr = f'god_map.closest_point.get_self_collisions(\'{self.link_a}\', \'{self.link_b}\')[{self.idx}].contact_distance'
        return symbol_manager.get_symbol(expr)

    def get_number_of_self_collisions(self):
        expr = f'god_map.closest_point.get_number_of_self_collisions(\'{self.link_a}\', \'{self.link_b}\')'
        return symbol_manager.get_symbol(expr)


class CollisionAvoidanceHint(Goal):
    def __init__(self, tip_link, avoidance_hint, object_link_name, object_group=None, max_linear_velocity=0.1,
                 root_link=None, max_threshold=0.05, spring_threshold=None, weight=WEIGHT_ABOVE_CA,
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        """
        This goal pushes the link_name in the direction of avoidance_hint, if it is closer than spring_threshold
        to body_b/link_b.
        :param tip_link: str, name of the robot link, has to have a collision body
        :param avoidance_hint: Vector3Stamped as json, direction in which the robot link will get pushed
        :param object_link_name: str, name of the link of the environment object. e.g. fridge handle
        :param max_linear_velocity: float, m/s, default 0.1
        :param root_link: str, default robot root, name of the root link for the kinematic chain
        :param max_threshold: float, default 0.05, distance at which the force has reached weight
        :param spring_threshold: float, default max_threshold, need to be >= than max_threshold weight increases from
                                        sprint_threshold to max_threshold linearly, to smooth motions
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        self.link_name = god_map.world.search_for_link_name(tip_link)
        self.link_b = god_map.world.search_for_link_name(object_link_name)
        if name is None:
            name = f'{self.__class__.__name__}/{self.link_name}/{self.link_b}'
        super().__init__(name)
        self.key = (self.link_name, self.link_b)
        self.object_group = object_group
        self.link_b_hash = self.link_b.__hash__()
        if root_link is None:
            self.root_link = god_map.world.root_link_name
        else:
            self.root_link = god_map.world.search_for_link_name(root_link)

        if spring_threshold is None:
            spring_threshold = max_threshold
        else:
            spring_threshold = max(spring_threshold, max_threshold)

        god_map.collision_scene.add_collision_check(god_map.world.links[self.link_name].name,
                                                    god_map.world.links[self.link_b].name,
                                                    spring_threshold)

        self.avoidance_hint = god_map.world.transform_msg(self.root_link, avoidance_hint)
        self.avoidance_hint.vector = tf.normalize(self.avoidance_hint.vector)

        self.max_velocity = max_linear_velocity
        self.threshold = max_threshold
        self.threshold2 = spring_threshold
        self.weight = weight
        actual_distance = self.get_actual_distance()
        max_velocity = self.max_velocity
        max_threshold = self.threshold
        spring_threshold = self.threshold2
        link_b_hash = self.get_link_b()
        actual_distance_capped = cas.max(actual_distance, 0)

        root_T_a = god_map.world.compose_fk_expression(self.root_link, self.link_name)

        spring_error = spring_threshold - actual_distance_capped
        spring_error = cas.max(spring_error, 0)

        spring_weight = cas.if_eq(spring_threshold, max_threshold, 0,
                                  weight * (spring_error / (spring_threshold - max_threshold)) ** 2)

        weight = cas.if_less_eq(actual_distance, max_threshold, weight,
                                spring_weight)
        weight = cas.if_eq(link_b_hash, self.link_b_hash, weight, 0)

        root_V_avoidance_hint = cas.Vector3(self.avoidance_hint)

        # penetration_distance = threshold - actual_distance_capped

        root_P_a = root_T_a.to_position()
        expr = root_V_avoidance_hint.dot(root_P_a)

        # self.add_debug_expr('dist', actual_distance)
        task = self.create_and_add_task('avoidance_hint')
        task.add_equality_constraint(reference_velocity=max_velocity,
                                     equality_bound=max_velocity,
                                     weight=weight,
                                     task_expression=expr)
        self.connect_monitors_to_all_tasks(start_condition, hold_condition, end_condition)

    def get_actual_distance(self):
        expr = f'god_map.closest_point.get_external_collisions_long_key(\'{self.key[0]}\', \'{self.key[1]}\').contact_distance'
        return symbol_manager.get_symbol(expr)

    def get_link_b(self):
        expr = f'god_map.closest_point.get_external_collisions_long_key(\'{self.key[0]}\', \'{self.key[1]}\').link_b_hash'
        return symbol_manager.get_symbol(expr)


# use cases
# avoid all
# allow all
# avoid all then allow something
# avoid only something

class CollisionAvoidance(Goal):
    def __init__(self,
                 collision_entries: List[CollisionEntry],
                 name: Optional[str] = None,
                 start_condition: cas.Expression = cas.TrueSymbol,
                 hold_condition: cas.Expression = cas.FalseSymbol,
                 end_condition: cas.Expression = cas.TrueSymbol):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name)
        self.start_condition = start_condition
        self.hold_condition = hold_condition
        self.end_condition = end_condition
        self.collision_matrix = god_map.collision_scene.create_collision_matrix(deepcopy(collision_entries))
        if not collision_entries or not god_map.collision_scene.is_allow_all_collision(collision_entries[-1]):
            self.add_external_collision_avoidance_constraints(
                soft_threshold_override=self.collision_matrix)
        if not collision_entries or (not god_map.collision_scene.is_allow_all_collision(collision_entries[-1]) and
                                     not god_map.collision_scene.is_allow_all_self_collision(collision_entries[-1])):
            self.add_self_collision_avoidance_constraints()
        if not cas.is_true(start_condition):
            payload_monitor = CollisionMatrixUpdater(name='update collision matrix',
                                                     start_condition=start_condition,
                                                     new_collision_matrix=self.collision_matrix)
            god_map.monitor_manager.add_payload_monitor(payload_monitor)
        else:
            god_map.collision_scene.set_collision_matrix(self.collision_matrix)

    def _task_sanity_check(self):
        pass

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
                robot_name = god_map.world.get_group_name_containing_link(child_link)
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
                    self.add_constraints_of_goal(ExternalCollisionAvoidance(robot_name=robot_name,
                                                                            link_name=child_link,
                                                                            name_prefix=self.name,
                                                                            hard_threshold=hard_threshold,
                                                                            soft_thresholds=soft_threshold,
                                                                            idx=i,
                                                                            num_repeller=number_of_repeller,
                                                                            start_condition=self.start_condition,
                                                                            hold_condition=self.hold_condition,
                                                                            end_condition=self.end_condition))
                    num_constrains += 1
        logging.loginfo(f'Adding {num_constrains} external collision avoidance constraints.')

    @profile
    def add_self_collision_avoidance_constraints(self):
        counter = defaultdict(int)
        fixed_joints = god_map.collision_scene.fixed_joints
        configs = god_map.collision_scene.collision_avoidance_configs
        num_constr = 0
        for robot_name in god_map.collision_scene.robot_names:
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
                groups_a = god_map.world.get_group_name_containing_link(link_a)
                groups_b = god_map.world.get_group_name_containing_link(link_b)
                if groups_b == groups_a:
                    robot_name = groups_a
                else:
                    raise Exception(f'Could not find group containing the link {link_a} and {link_b}.')
                self.add_constraints_of_goal(SelfCollisionAvoidance(link_a=link_a,
                                                                    link_b=link_b,
                                                                    robot_name=robot_name,
                                                                    name_prefix=self.name,
                                                                    hard_threshold=hard_threshold,
                                                                    soft_threshold=soft_threshold,
                                                                    idx=i,
                                                                    num_repeller=number_of_repeller,
                                                                    start_condition=self.start_condition,
                                                                    hold_condition=self.hold_condition,
                                                                    end_condition=self.end_condition))
                num_constr += 1
        logging.loginfo(f'Adding {num_constr} self collision avoidance constraints.')
