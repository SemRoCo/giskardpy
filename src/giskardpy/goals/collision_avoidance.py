from typing import Dict, Optional

import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w, identifier
from giskardpy.goals.goal import Goal, WEIGHT_COLLISION_AVOIDANCE, WEIGHT_ABOVE_CA
from giskardpy.my_types import my_string


class ExternalCollisionAvoidance(Goal):

    def __init__(self,
                 link_name: my_string,
                 robot_name: str,
                 max_velocity: float = 0.2,
                 hard_threshold: float = 0.0,
                 soft_thresholds: Optional[Dict[my_string, float]] = None,
                 idx: int = 0,
                 num_repeller: int = 1):
        """
        Don't use me
        """
        self.max_velocity = max_velocity
        self.hard_threshold = hard_threshold
        self.soft_thresholds = soft_thresholds
        self.num_repeller = num_repeller
        self.link_name = link_name
        self.idx = idx
        super().__init__()
        self.root = self.world.root_link_name
        self.robot_name = robot_name
        self.control_horizon = self.prediction_horizon - (self.god_map.get_data(identifier.max_derivative) - 1)
        self.control_horizon = max(1, self.control_horizon)

    # def get_contact_normal_on_b_in_root(self):
    #     return self.god_map.list_to_vector3(identifier.closest_point + ['get_external_collisions',
    #                                                                           (self.link_name,),
    #                                                                           self.idx,
    #                                                                           'root_V_n'])

    def map_V_n_symbol(self):
        return self.god_map.list_to_vector3(identifier.closest_point + ['get_external_collisions',
                                                                        (self.link_name,),
                                                                        self.idx,
                                                                        'map_V_n'])

    def get_closest_point_on_a_in_a(self):
        return self.god_map.list_to_point3(identifier.closest_point + ['get_external_collisions',
                                                                       (self.link_name,),
                                                                       self.idx,
                                                                       'new_a_P_pa'])

    def map_P_a_symbol(self):
        return self.god_map.list_to_point3(identifier.closest_point + ['get_external_collisions',
                                                                       (self.link_name,),
                                                                       self.idx,
                                                                       'new_map_P_pa'])

    # def get_closest_point_on_b_in_root(self):
    #     return self.god_map.list_to_point3(identifier.closest_point + ['get_external_collisions',
    #                                                                          (self.link_name,),
    #                                                                          self.idx,
    #                                                                          'root_P_b'])

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_external_collisions',
                                                                  (self.link_name,),
                                                                  self.idx,
                                                                  'contact_distance'])

    def get_link_b_hash(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_external_collisions',
                                                                  (self.link_name,),
                                                                  self.idx,
                                                                  'link_b_hash'])

    def get_number_of_external_collisions(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_number_of_external_collisions',
                                                                  (self.link_name,)])

    @profile
    def make_constraints(self):
        a_P_pa = self.get_closest_point_on_a_in_a()
        map_V_n = self.map_V_n_symbol()
        actual_distance = self.get_actual_distance()
        sample_period = self.sample_period
        number_of_external_collisions = self.get_number_of_external_collisions()

        map_T_a = self.get_fk(self.root, self.link_name)

        map_P_pa = map_T_a.dot(a_P_pa)

        # the position distance is not accurate, but the derivative is still correct
        dist = map_V_n.dot(map_P_pa)

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        soft_threshold = 0
        actual_link_b_hash = self.get_link_b_hash()
        parent_joint = self.world.links[self.link_name].parent_joint_name
        direct_children = set(self.world.get_directly_controlled_child_links_with_collisions(parent_joint))
        b_result_cases = [(k[1].__hash__(), v) for k, v in self.soft_thresholds.items() if k[0] in direct_children]
        soft_threshold = w.if_eq_cases(a=actual_link_b_hash,
                                       b_result_cases=b_result_cases,
                                       else_result=soft_threshold)

        hard_threshold = w.min(self.hard_threshold, soft_threshold / 2)
        lower_limit = soft_threshold - actual_distance

        lower_limit_limited = w.limit(lower_limit,
                                      -qp_limits_for_lba,
                                      qp_limits_for_lba)

        upper_slack = w.if_greater(actual_distance, hard_threshold,
                                   w.limit(soft_threshold - hard_threshold,
                                           -qp_limits_for_lba,
                                           qp_limits_for_lba),
                                   lower_limit_limited)
        # undo factor in A
        upper_slack /= (sample_period * self.control_horizon)

        upper_slack = w.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                   1e4,
                                   w.max(0, upper_slack))

        weight = w.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)

        weight = w.save_division(weight,  # divide by number of active repeller per link
                                 w.min(number_of_external_collisions, self.num_repeller))
        # if self.link_name == 'base_footprint' and self.idx <= 1:
        #     self.add_debug_expr('soft_threshold', soft_threshold)
        #     self.add_debug_expr('weight', weight)
        #     self.add_debug_expr('soft_threshold', soft_threshold)
        #     self.add_debug_expr('dist', dist)
        #     self.add_debug_expr('actual_distance', actual_distance)
        self.add_inequality_constraint(reference_velocity=self.max_velocity,
                                       lower_error=lower_limit,
                                       upper_error=float('inf'),
                                       weight=weight,
                                       task_expression=dist,
                                       lower_slack_limit=-float('inf'),
                                       upper_slack_limit=upper_slack)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.link_name}/{self.idx}'


class SelfCollisionAvoidance(Goal):

    def __init__(self,
                 link_a: my_string,
                 link_b: my_string,
                 robot_name: str,
                 max_velocity: float = 0.2,
                 hard_threshold: float = 0.0,
                 soft_threshold: float = 0.05,
                 idx: float = 0,
                 num_repeller: int = 1):
        self.link_a = link_a
        self.link_b = link_b
        self.max_velocity = max_velocity
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.num_repeller = num_repeller
        self.idx = idx
        if self.link_a.prefix != self.link_b.prefix:
            raise Exception(f'Links {self.link_a} and {self.link_b} have different prefix.')
        super().__init__()
        self.root = self.world.root_link_name
        self.robot_name = robot_name
        self.control_horizon = self.prediction_horizon - (self.god_map.get_data(identifier.max_derivative) - 1)
        self.control_horizon = max(1, self.control_horizon)

    def get_contact_normal_in_b(self):
        return self.god_map.list_to_vector3(identifier.closest_point + ['get_self_collisions',
                                                                        (self.link_a, self.link_b),
                                                                        self.idx,
                                                                        'new_b_V_n'])

    def get_position_on_a_in_a(self):
        return self.god_map.list_to_point3(identifier.closest_point + ['get_self_collisions',
                                                                       (self.link_a, self.link_b),
                                                                       self.idx,
                                                                       'new_a_P_pa'])

    def get_b_T_pb(self) -> w.TransMatrix:
        return self.god_map.list_to_translation3(identifier.closest_point + ['get_self_collisions',
                                                                             (self.link_a, self.link_b),
                                                                             self.idx,
                                                                             'new_b_P_pb'])

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_self_collisions',
                                                                  (self.link_a, self.link_b),
                                                                  self.idx,
                                                                  'contact_distance'])

    def get_number_of_self_collisions(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_number_of_self_collisions',
                                                                  (self.link_a, self.link_b)])

    @profile
    def make_constraints(self):
        hard_threshold = w.min(self.hard_threshold, self.soft_threshold / 2)
        actual_distance = self.get_actual_distance()
        number_of_self_collisions = self.get_number_of_self_collisions()
        sample_period = self.sample_period

        # b_T_a2 = self.get_fk_evaluated(self.link_b, self.link_a)
        b_T_a = self.get_fk(self.link_b, self.link_a)
        pb_T_b = self.get_b_T_pb().inverse()
        a_P_pa = self.get_position_on_a_in_a()

        pb_V_n = self.get_contact_normal_in_b()

        pb_P_pa = pb_T_b.dot(b_T_a).dot(a_P_pa)

        dist = pb_V_n.dot(pb_P_pa)

        weight = w.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)
        weight = w.save_division(weight,  # divide by number of active repeller per link
                                 w.min(number_of_self_collisions, self.num_repeller))

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        lower_limit = self.soft_threshold - actual_distance

        lower_limit_limited = w.limit(lower_limit,
                                      -qp_limits_for_lba,
                                      qp_limits_for_lba)

        upper_slack = w.if_greater(actual_distance, hard_threshold,
                                   w.limit(self.soft_threshold - hard_threshold,
                                           -qp_limits_for_lba,
                                           qp_limits_for_lba),
                                   lower_limit_limited)

        # undo factor in A
        upper_slack /= (sample_period * self.control_horizon)

        upper_slack = w.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                   1e4,
                                   w.max(0, upper_slack))

        self.add_inequality_constraint(reference_velocity=self.max_velocity,
                                       lower_error=lower_limit,
                                       upper_error=float('inf'),
                                       weight=weight,
                                       task_expression=dist,
                                       lower_slack_limit=-float('inf'),
                                       upper_slack_limit=upper_slack)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.link_a}/{self.link_b}/{self.idx}'


class CollisionAvoidanceHint(Goal):
    def __init__(self, tip_link, avoidance_hint, object_link_name, object_group=None, max_linear_velocity=0.1,
                 root_link=None, max_threshold=0.05, spring_threshold=None, weight=WEIGHT_ABOVE_CA):
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
        super().__init__()
        self.link_name = self.world.search_for_link_name(tip_link)
        self.link_b = self.world.search_for_link_name(object_link_name)
        self.key = (self.link_name, self.link_b)
        self.object_group = object_group
        self.link_b_hash = self.link_b.__hash__()
        if root_link is None:
            self.root_link = self.world.root_link_name
        else:
            self.root_link = self.world.search_for_link_name(root_link)

        if spring_threshold is None:
            spring_threshold = max_threshold
        else:
            spring_threshold = max(spring_threshold, max_threshold)

        self.add_collision_check(self.world.links[self.link_name].name,
                                 self.world.links[self.link_b].name,
                                 spring_threshold)

        self.avoidance_hint = self.world.transform_msg(self.root_link, avoidance_hint)
        self.avoidance_hint.vector = tf.normalize(self.avoidance_hint.vector)

        self.max_velocity = max_linear_velocity
        self.threshold = max_threshold
        self.threshold2 = spring_threshold
        self.weight = weight

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_external_collisions_long_key',
                                                                  self.key,
                                                                  'contact_distance'])

    def get_link_b(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_external_collisions_long_key',
                                                                  self.key,
                                                                  'link_b_hash'])

    def make_constraints(self):
        weight = self.weight
        actual_distance = self.get_actual_distance()
        max_velocity = self.max_velocity
        max_threshold = self.threshold
        spring_threshold = self.threshold2
        link_b_hash = self.get_link_b()
        actual_distance_capped = w.max(actual_distance, 0)

        root_T_a = self.get_fk(self.root_link, self.link_name)

        spring_error = spring_threshold - actual_distance_capped
        spring_error = w.max(spring_error, 0)

        spring_weight = w.if_eq(spring_threshold, max_threshold, 0,
                                weight * (spring_error / (spring_threshold - max_threshold)) ** 2)

        weight = w.if_less_eq(actual_distance, max_threshold, weight,
                              spring_weight)
        weight = w.if_eq(link_b_hash, self.link_b_hash, weight, 0)

        root_V_avoidance_hint = w.Vector3(self.avoidance_hint)

        # penetration_distance = threshold - actual_distance_capped

        root_P_a = root_T_a.to_position()
        expr = root_V_avoidance_hint.dot(root_P_a)

        # self.add_debug_expr('dist', actual_distance)
        self.add_equality_constraint(name='avoidance_hint',
                                     reference_velocity=max_velocity,
                                     equality_bound=max_velocity,
                                     weight=weight,
                                     task_expression=expr)

    def __str__(self):
        s = super().__str__()
        return f'{s}/{self.link_name}/{self.link_b}'
