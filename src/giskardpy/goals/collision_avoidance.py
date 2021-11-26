import giskardpy.utils.tfwrapper as tf
from giskardpy import casadi_wrapper as w, identifier, RobotName
from giskardpy.goals.goal import Goal, WEIGHT_COLLISION_AVOIDANCE, WEIGHT_ABOVE_CA


class ExternalCollisionAvoidance(Goal):

    def __init__(self, link_name, max_velocity=0.2, hard_threshold=0.0, soft_threshold=0.05, idx=0,
                 num_repeller=1, **kwargs):
        """
        Don't use me
        """
        self.max_velocity = max_velocity
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.num_repeller = num_repeller
        self.link_name = link_name
        self.idx = idx
        super(ExternalCollisionAvoidance, self).__init__(**kwargs)
        self.root = self.world.root_link_name
        self.robot_name = self.robot.name

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

    def get_number_of_external_collisions(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_number_of_external_collisions',
                                                                  (self.link_name,)])

    def make_constraints(self):
        a_P_pa = self.get_closest_point_on_a_in_a()
        map_V_n = self.map_V_n_symbol()
        actual_distance = self.get_actual_distance()
        sample_period = self.get_sampling_period_symbol()
        number_of_external_collisions = self.get_number_of_external_collisions()

        map_T_a = self.get_fk(self.root, self.link_name)

        map_P_pa = w.dot(map_T_a, a_P_pa)

        dist = w.dot(map_V_n.T, map_P_pa)[0]

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        lower_limit = self.soft_threshold - actual_distance

        lower_limit_limited = w.limit(lower_limit,
                                      -qp_limits_for_lba,
                                      qp_limits_for_lba)

        upper_slack = w.if_greater(actual_distance, self.hard_threshold,
                                   w.limit(self.soft_threshold - self.hard_threshold,
                                           -qp_limits_for_lba,
                                           qp_limits_for_lba),
                                   lower_limit_limited)

        # undo factor in A
        upper_slack /= (sample_period * self.prediction_horizon)

        upper_slack = w.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                   1e4,
                                   w.max(0, upper_slack))

        weight = w.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)

        weight = w.save_division(weight,  # divide by number of active repeller per link
                                 w.min(number_of_external_collisions, self.num_repeller))

        self.add_constraint(reference_velocity=self.max_velocity,
                            lower_error=lower_limit,
                            upper_error=100,
                            weight=weight,
                            expression=dist,
                            lower_slack_limit=-1e4,
                            upper_slack_limit=upper_slack)

    def __str__(self):
        s = super(ExternalCollisionAvoidance, self).__str__()
        return '{}/{}/{}'.format(s, self.link_name, self.idx)


class SelfCollisionAvoidance(Goal):

    def __init__(self, link_a, link_b, max_velocity=0.2, hard_threshold=0.0, soft_threshold=0.05, idx=0,
                 num_repeller=1, **kwargs):
        self.link_a = link_a
        self.link_b = link_b
        self.max_velocity = max_velocity
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.num_repeller = num_repeller
        self.idx = idx
        super(SelfCollisionAvoidance, self).__init__(**kwargs)
        self.root = self.world.root_link_name
        self.robot_name = RobotName

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

    def get_b_T_pb(self):
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

    def make_constraints(self):
        actual_distance = self.get_actual_distance()
        number_of_self_collisions = self.get_number_of_self_collisions()
        sample_period = self.get_sampling_period_symbol()

        b_T_a2 = self.get_fk_evaluated(self.link_b, self.link_a)
        b_T_a = self.get_fk(self.link_b, self.link_a)
        pb_T_b = w.inverse_frame(self.get_b_T_pb())
        a_P_pa = self.get_position_on_a_in_a()

        pb_V_n = self.get_contact_normal_in_b()

        pb_P_pa = w.dot(pb_T_b, b_T_a, a_P_pa)

        dist = w.dot(pb_V_n.T, pb_P_pa)[0]

        weight = w.if_greater(actual_distance, 50, 0, WEIGHT_COLLISION_AVOIDANCE)
        weight = w.save_division(weight,  # divide by number of active repeller per link
                                 w.min(number_of_self_collisions, self.num_repeller))

        qp_limits_for_lba = self.max_velocity * sample_period * self.control_horizon

        lower_limit = self.soft_threshold - actual_distance

        lower_limit_limited = w.limit(lower_limit,
                                      -qp_limits_for_lba,
                                      qp_limits_for_lba)

        upper_slack = w.if_greater(actual_distance, self.hard_threshold,
                                   w.limit(self.soft_threshold - self.hard_threshold,
                                           -qp_limits_for_lba,
                                           qp_limits_for_lba),
                                   lower_limit_limited)

        # undo factor in A
        upper_slack /= (sample_period * self.prediction_horizon)

        upper_slack = w.if_greater(actual_distance, 50,  # assuming that distance of unchecked closest points is 100
                                   1e4,
                                   w.max(0, upper_slack))

        self.add_constraint(reference_velocity=self.max_velocity,
                            lower_error=lower_limit,
                            upper_error=100,
                            weight=weight,
                            expression=dist,
                            lower_slack_limit=-1e4,
                            upper_slack_limit=upper_slack)

    def __str__(self):
        s = super(SelfCollisionAvoidance, self).__str__()
        return '{}/{}/{}/{}'.format(s, self.link_a, self.link_b, self.idx)


class CollisionAvoidanceHint(Goal):
    def __init__(self, tip_link, avoidance_hint, object_name, object_link_name, max_linear_velocity=0.1,
                 root_link=None, max_threshold=0.05, spring_threshold=None, weight=WEIGHT_ABOVE_CA, **kwargs):
        """
        This goal pushes the link_name in the direction of avoidance_hint, if it is closer than spring_threshold
        to body_b/link_b.
        :param tip_link: str, name of the robot link, has to have a collision body
        :param avoidance_hint: Vector3Stamped as json, direction in which the robot link will get pushed
        :param object_name: str, name of the environment object, can be the robot, e.g. kitchen
        :param object_link_name: str, name of the link of the environment object. e.g. fridge handle
        :param max_linear_velocity: float, m/s, default 0.1
        :param root_link: str, default robot root, name of the root link for the kinematic chain
        :param max_threshold: float, default 0.05, distance at which the force has reached weight
        :param spring_threshold: float, default max_threshold, need to be >= than max_threshold weight increases from
                                        sprint_threshold to max_threshold linearly, to smooth motions
        :param weight: float, default WEIGHT_ABOVE_CA
        """
        super(CollisionAvoidanceHint, self).__init__(**kwargs)
        self.link_name = tip_link
        self.key = (tip_link, object_name, object_link_name)
        self.body_b = object_name
        self.link_b = object_link_name
        self.body_b_hash = object_name.__hash__()
        self.link_b_hash = object_link_name.__hash__()
        if root_link is None:
            self.root_link = self.robot.root_link_name
        else:
            self.root_link = root_link

        if spring_threshold is None:
            spring_threshold = max_threshold
        else:
            spring_threshold = max(spring_threshold, max_threshold)

        # register collision checks TODO make function
        added_checks = self.god_map.get_data(identifier.added_collision_checks)
        if tip_link in added_checks:
            added_checks[tip_link] = max(added_checks[tip_link], spring_threshold)
        else:
            added_checks[tip_link] = spring_threshold
        self.god_map.set_data(identifier.added_collision_checks, added_checks)

        self.avoidance_hint = tf.transform_vector(self.root_link, avoidance_hint)
        self.avoidance_hint.vector = tf.normalize(self.avoidance_hint.vector)

        self.max_velocity = max_linear_velocity
        self.threshold = max_threshold
        self.threshold2 = spring_threshold
        self.weight = weight

    def get_actual_distance(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_external_collisions_long_key',
                                                                  self.key,
                                                                  'contact_distance'])

    def get_body_b(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_external_collisions_long_key',
                                                                  self.key, 'get_body_b_hash', tuple()])

    def get_link_b(self):
        return self.god_map.to_symbol(identifier.closest_point + ['get_external_collisions_long_key',
                                                                  self.key, 'get_link_b_hash', tuple()])

    def make_constraints(self):
        weight = self.get_parameter_as_symbolic_expression('weight')
        actual_distance = self.get_actual_distance()
        max_velocity = self.get_parameter_as_symbolic_expression('max_velocity')
        max_threshold = self.get_parameter_as_symbolic_expression('threshold')
        spring_threshold = self.get_parameter_as_symbolic_expression('threshold2')
        body_b_hash = self.get_body_b()
        link_b_hash = self.get_link_b()
        actual_distance_capped = w.max(actual_distance, 0)

        root_T_a = self.get_fk(self.root_link, self.link_name)

        spring_error = spring_threshold - actual_distance_capped
        spring_error = w.max(spring_error, 0)

        spring_weight = w.if_eq(spring_threshold, max_threshold, 0,
                                weight * (spring_error / (spring_threshold - max_threshold)) ** 2)

        weight = w.if_less_eq(actual_distance, max_threshold, weight,
                              spring_weight)
        weight = w.if_eq(body_b_hash, self.body_b_hash, weight, 0)
        weight = w.if_eq(link_b_hash, self.link_b_hash, weight, 0)

        root_V_avoidance_hint = self.get_parameter_as_symbolic_expression('avoidance_hint')

        # penetration_distance = threshold - actual_distance_capped

        root_P_a = w.position_of(root_T_a)
        expr = w.dot(root_V_avoidance_hint[:3].T, root_P_a[:3])

        # FIXME really?
        self.add_constraint(name_suffix='avoidance_hint',
                            reference_velocity=max_velocity,
                            lower_error=max_velocity,
                            upper_error=max_velocity,
                            weight=weight,
                            expression=expr)

    def __str__(self):
        s = super(CollisionAvoidanceHint, self).__str__()
        return '{}/{}/{}/{}'.format(s, self.link_name, self.body_b, self.link_b)
